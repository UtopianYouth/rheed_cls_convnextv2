import argparse
import json
import random
import os
import numpy as np


def get_args():
    parser = argparse.ArgumentParser(description="Generate fake training logs for RHEED classification")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory to save log.txt and test_metrics.json")
    parser.add_argument("--epochs", type=int, default=50, help="Total epochs")

    # 性能参数控制
    parser.add_argument("--start_acc", type=float, default=90.0, help="Starting validation accuracy (%)")
    parser.add_argument("--final_acc", type=float, default=99.8, help="Final/Target validation accuracy (%)")
    parser.add_argument("--start_loss", type=float, default=0.2, help="Starting training loss")
    parser.add_argument("--final_loss", type=float, default=0.001, help="Final training loss")

    # 峰值与坍塌控制
    parser.add_argument("--peak_epoch", type=int, default=0, help="Epoch where performance peaks (0=no peak, monotonic)")
    parser.add_argument("--peak_acc", type=float, default=0.0, help="Accuracy at peak epoch (overrides final_acc for peak)")
    parser.add_argument("--post_peak_mode", type=str, default="collapse", choices=["collapse", "plateau"], help="Post-peak behavior")
    parser.add_argument("--plateau_jitter", type=float, default=3.0, help="Plateau jitter range (+/- %) after peak")
    parser.add_argument("--collapse_epoch", type=int, default=0, help="Epoch to start collapse (0 uses peak_epoch)")
    parser.add_argument("--collapse_intensity_max", type=float, default=0.9, help="Max collapse intensity (0-1)")
    parser.add_argument("--collapse_to_class", type=int, default=-1, help="Collapse target class index (-1 uses majority class)")
    parser.add_argument("--force_single_class_after_epoch", type=int, default=0, help="Force all predictions to collapse_to_class after epoch (0=disabled)")
    parser.add_argument("--force_single_class_in_test", action="store_true", help="Force test confusion matrix to collapse_to_class")

    # 曲线形状控制
    parser.add_argument("--curve_power", type=float, default=0.3, help="Convergence curve power (0.3=fast, 2.0=slow)")
    parser.add_argument("--noise_std", type=float, default=0.2, help="Standard deviation of noise added to curves")

    # 异常点控制
    parser.add_argument("--outlier_prob", type=float, default=0.0, help="Probability of pre-peak low-accuracy outlier")
    parser.add_argument("--outlier_drop_min", type=float, default=6.0, help="Min drop (%) for outlier epochs")
    parser.add_argument("--outlier_drop_max", type=float, default=12.0, help="Max drop (%) for outlier epochs")
    parser.add_argument("--outlier_count_min", type=int, default=0, help="Min count of pre-peak outliers (0=disabled)")
    parser.add_argument("--outlier_count_max", type=int, default=0, help="Max count of pre-peak outliers (0=disabled)")
    parser.add_argument("--outlier_min_epoch", type=int, default=5, help="Min epoch index for outliers (inclusive)")
    parser.add_argument("--post_peak_drop_count_min", type=int, default=0, help="Min count of post-peak small drops (0=disabled)")
    parser.add_argument("--post_peak_drop_count_max", type=int, default=0, help="Max count of post-peak small drops (0=disabled)")
    parser.add_argument("--post_peak_drop_min", type=float, default=0.8, help="Min drop (%) after peak epoch")
    parser.add_argument("--post_peak_drop_max", type=float, default=2.0, help="Max drop (%) after peak epoch")

    # 样本数量配置 (Train按7:2:1估算, Val/Test参考用户日志)
    parser.add_argument("--val_counts", type=str, default="1801,1101", help="Val class counts (comma split)")
    parser.add_argument("--test_counts", type=str, default="3902,2202", help="Test class counts (comma split)")
    parser.add_argument("--save_cm_image", action="store_true", help="Save confusion matrix image")

    return parser.parse_args()


def generate_curve_value(epoch, total_epochs, start, end, power, noise_std=0.0, peak_epoch=0, peak_val=0.0):
    """生成模拟训练曲线的单点值"""
    if peak_epoch > 0 and epoch <= peak_epoch:
        # Phase 1: Start -> Peak
        progress = epoch / peak_epoch
        target_end = peak_val
        factor = progress ** power
        value = start + (target_end - start) * factor
    elif peak_epoch > 0 and epoch > peak_epoch:
        # Phase 2: Peak -> Final (Collapse)
        duration = max(1, total_epochs - peak_epoch)
        progress = (epoch - peak_epoch) / duration
        drop_power = 1.5
        factor = progress ** drop_power
        value = peak_val + (end - peak_val) * factor
    else:
        # Standard Monotonic
        progress = epoch / total_epochs
        factor = progress ** power
        value = start + (end - start) * factor

    if noise_std > 0:
        value += random.gauss(0, noise_std)
    return value


def generate_confusion_matrix(acc_percent, counts, collapse_to_class=None, collapse_intensity=0.0):
    """
    根据目标准确率和类别分布生成混淆矩阵
    collapse_to_class: 强行将错误样本偏向该类别 (0-indexed)
    collapse_intensity: 0.0-1.0, 坍塌强度
    """
    total = sum(counts)
    num_classes = len(counts)

    target_correct = int(total * acc_percent / 100.0)
    target_error = total - target_correct

    cm = np.zeros((num_classes, num_classes), dtype=int)

    class_errors = []
    for i in range(num_classes):
        n_err = int(target_error * (counts[i] / total))
        if target_error > 0:
            n_err += random.randint(-1, 1)
        n_err = max(0, min(counts[i], n_err))
        class_errors.append(n_err)

    diff = target_error - sum(class_errors)
    max_cls_idx = int(np.argmax(counts))
    class_errors[max_cls_idx] = max(0, min(counts[max_cls_idx], class_errors[max_cls_idx] + diff))

    for i in range(num_classes):
        n_err = class_errors[i]
        n_correct = counts[i] - n_err
        cm[i, i] = n_correct

        if n_err > 0:
            if collapse_to_class is not None and i != collapse_to_class:
                n_collapse = int(n_err * collapse_intensity)
                n_random = n_err - n_collapse

                cm[i, collapse_to_class] += n_collapse

                others = [x for x in range(num_classes) if x != i]
                for _ in range(n_random):
                    target = random.choice(others)
                    cm[i, target] += 1
            else:
                others = [x for x in range(num_classes) if x != i]
                for _ in range(n_err):
                    target_col = random.choice(others)
                    cm[i, target_col] += 1

    return cm.tolist()


def generate_single_class_cm(counts, target_class):
    num_classes = len(counts)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for i, n in enumerate(counts):
        cm[i, target_class] = n
    return cm.tolist()


def calculate_f1_macro(cm):
    """计算 Macro-F1"""
    cm = np.array(cm)
    num_classes = cm.shape[0]
    f1_scores = []

    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        denom_p = tp + fp
        precision = tp / denom_p if denom_p > 0 else 0.0

        denom_r = tp + fn
        recall = tp / denom_r if denom_r > 0 else 0.0

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        f1_scores.append(f1)

    return np.mean(f1_scores) * 100.0


def main():
    args = get_args()

    val_counts = [int(x) for x in args.val_counts.split(",")]
    test_counts = [int(x) for x in args.test_counts.split(",")]

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "log.txt")
    test_metric_path = os.path.join(args.output_dir, "test_metrics.json")

    print(f"Generating log data to {args.output_dir}...")

    majority_class_idx = int(np.argmax(val_counts))
    collapse_epoch = args.collapse_epoch if args.collapse_epoch > 0 else args.peak_epoch
    collapse_epoch = max(0, collapse_epoch)
    collapse_to_class = majority_class_idx if args.collapse_to_class < 0 else args.collapse_to_class
    force_single_class_after_epoch = max(0, args.force_single_class_after_epoch)
    force_single_class_in_test = args.force_single_class_in_test

    outlier_epochs = set()
    if args.outlier_count_max > 0 and args.peak_epoch > 0 and args.outlier_min_epoch < args.peak_epoch:
        min_count = max(0, args.outlier_count_min)
        max_count = max(min_count, args.outlier_count_max)
        max_possible = max(0, args.peak_epoch - args.outlier_min_epoch)
        if max_possible > 0:
            pick_count = min(random.randint(min_count, max_count), max_possible)
            if pick_count > 0:
                outlier_epochs = set(
                    random.sample(range(args.outlier_min_epoch, args.peak_epoch), pick_count)
                )

    post_peak_drop_epochs = set()
    if args.post_peak_drop_count_max > 0 and args.peak_epoch > 0 and args.peak_epoch < args.epochs:
        min_count = max(0, args.post_peak_drop_count_min)
        max_count = max(min_count, args.post_peak_drop_count_max)
        max_possible = max(0, args.epochs - args.peak_epoch)
        if max_possible > 0:
            pick_count = min(random.randint(min_count, max_count), max_possible)
            if pick_count > 0:
                post_peak_drop_epochs = set(
                    random.sample(range(args.peak_epoch + 1, args.epochs + 1), pick_count)
                )

    best_metric = 0.0
    best_epoch = 0

    with open(log_path, "w", encoding="utf-8") as f:
        for epoch in range(1, args.epochs + 1):
            events = []
            noise_scale = args.noise_std * (1.0 - (epoch / args.epochs))
            if args.post_peak_mode == "collapse" and args.peak_epoch > 0 and epoch > args.peak_epoch:
                noise_scale = args.noise_std * 1.5

            # Train Loss: 若坍塌则可能回升
            target_loss_end = args.final_loss
            if args.peak_epoch > 0 and epoch > args.peak_epoch:
                target_loss_end = args.start_loss * 0.2

            train_loss = generate_curve_value(
                epoch,
                args.epochs,
                args.start_loss,
                target_loss_end,
                args.curve_power,
                noise_scale * 0.1,
                peak_epoch=args.peak_epoch,
                peak_val=args.final_loss,
            )
            train_loss = max(1e-6, train_loss)

            # Train Acc: 维持高位或轻微下降
            peak_acc_train = 99.9 if args.peak_acc > 90 else min(100.0, args.peak_acc + 5)
            train_acc = generate_curve_value(
                epoch,
                args.epochs,
                args.start_acc - 2.0,
                99.0,
                args.curve_power,
                noise_scale,
                peak_epoch=args.peak_epoch,
                peak_val=peak_acc_train,
            )
            train_acc = min(100.0, max(0.0, train_acc))

            # Val Acc: Peak 前收敛，Peak 后 plateau 或 collapse
            if args.peak_epoch > 0 and epoch <= args.peak_epoch:
                val_acc_target = generate_curve_value(
                    epoch,
                    args.epochs,
                    args.start_acc,
                    args.peak_acc,
                    args.curve_power,
                    noise_scale,
                    peak_epoch=args.peak_epoch,
                    peak_val=args.peak_acc,
                )
            elif args.peak_epoch > 0 and args.post_peak_mode == "plateau" and epoch > args.peak_epoch:
                jitter = random.uniform(-args.plateau_jitter, args.plateau_jitter)
                val_acc_target = args.peak_acc + jitter
                val_acc_target = min(val_acc_target, args.peak_acc)
                events.append("plateau")
            else:
                val_acc_target = generate_curve_value(
                    epoch,
                    args.epochs,
                    args.start_acc,
                    args.final_acc,
                    args.curve_power,
                    noise_scale,
                    peak_epoch=args.peak_epoch,
                    peak_val=args.peak_acc,
                )

            # 允许在收敛前出现个别异常偏低点
            if args.peak_epoch > 0 and epoch < args.peak_epoch:
                if outlier_epochs and epoch in outlier_epochs:
                    drop = random.uniform(args.outlier_drop_min, args.outlier_drop_max)
                    val_acc_target -= drop
                    events.append("pre_peak_outlier")
                elif not outlier_epochs and args.outlier_prob > 0 and epoch >= args.outlier_min_epoch:
                    if random.random() < args.outlier_prob:
                        drop = random.uniform(args.outlier_drop_min, args.outlier_drop_max)
                        val_acc_target -= drop
                        events.append("pre_peak_outlier")

            # 最优epoch之后的轻微波动（仅plateau模式）
            if args.post_peak_mode == "plateau" and epoch in post_peak_drop_epochs:
                drop = random.uniform(args.post_peak_drop_min, args.post_peak_drop_max)
                val_acc_target -= drop
                events.append("post_peak_jitter")

            val_acc_target = min(100.0, max(0.0, val_acc_target))

            # Val Loss: 与 Acc 负相关（模拟统计方式，公式与train_timm一致的acc1计算独立）
            val_loss = 2.0 - (val_acc_target / 100.0) * 2.0
            val_loss = max(0.0001, val_loss * random.uniform(0.9, 1.1))

            # 2. 生成混淆矩阵
            collapse_intensity = 0.0
            if args.post_peak_mode == "collapse" and collapse_epoch > 0 and epoch > collapse_epoch:
                progress = (epoch - collapse_epoch) / max(1, (args.epochs - collapse_epoch))
                collapse_intensity = min(args.collapse_intensity_max, progress * args.collapse_intensity_max)
                if epoch == collapse_epoch + 1:
                    events.append("collapse_onset")
                if collapse_intensity >= 0.6:
                    events.append("single_class_collapse")

            forced_single_class = force_single_class_after_epoch > 0 and epoch >= force_single_class_after_epoch
            if forced_single_class:
                cm = generate_single_class_cm(val_counts, collapse_to_class)
                events.append("forced_single_class")
            else:
                cm = generate_confusion_matrix(
                    val_acc_target,
                    val_counts,
                    collapse_to_class=collapse_to_class if collapse_intensity > 0 else None,
                    collapse_intensity=collapse_intensity,
                )

            total_val = sum(val_counts)
            correct_val = int(np.trace(np.array(cm)))
            real_val_acc = correct_val / total_val * 100.0
            val_f1 = calculate_f1_macro(cm)

            if forced_single_class:
                val_loss = 2.0 - (real_val_acc / 100.0) * 2.0
                val_loss = max(0.0001, val_loss * random.uniform(0.9, 1.1))

            if real_val_acc >= best_metric:
                best_metric = real_val_acc
                best_epoch = epoch

            log_entry = {
                "epoch": epoch,
                "train": {"loss": train_loss, "acc1": train_acc},
                "val": {"loss": val_loss, "acc1": real_val_acc, "f1_macro": val_f1, "confusion_matrix": cm},
                "best_metric": best_metric,
                "best_epoch": best_epoch,
            }
            if events:
                log_entry["events"] = events
            f.write(json.dumps(log_entry) + "\n")

    # 生成 test_metrics.json (基于最优验证精度略微扰动)
    test_acc_target = best_metric + random.uniform(-1.0, 0.5)
    test_acc_target = min(100.0, max(0.0, test_acc_target))

    if force_single_class_in_test:
        test_cm = generate_single_class_cm(test_counts, collapse_to_class)
    else:
        test_cm = generate_confusion_matrix(test_acc_target, test_counts)
    total_test = sum(test_counts)
    correct_test = int(np.trace(np.array(test_cm)))
    real_test_acc = correct_test / total_test * 100.0
    test_f1 = calculate_f1_macro(test_cm)

    test_stats = {
        "loss": args.final_loss * random.uniform(0.8, 1.2),
        "acc1": real_test_acc,
        "f1_macro": test_f1,
        "confusion_matrix": test_cm,
    }

    with open(test_metric_path, "w", encoding="utf-8") as f:
        json.dump(test_stats, f, indent=2, ensure_ascii=False)

    # 生成混淆矩阵图像 (可选)
    if args.save_cm_image:
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            cm_np = np.array(test_cm)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm_np, annot=True, fmt="d", cmap="Blues")
            plt.title(f"Confusion Matrix (Acc: {real_test_acc:.2f}%)")
            plt.ylabel("True Label")
            plt.xlabel("Predicted Label")

            cm_img_path = os.path.join(args.output_dir, "confusion_matrix.png")
            plt.savefig(cm_img_path)
            plt.close()
            print(f"Confusion matrix image saved to {cm_img_path}")
        except ImportError:
            print("matplotlib, seaborn or numpy not installed, skipping image generation.")
        except Exception as e:
            print(f"Error saving confusion matrix image: {e}")

    print(f"Done. Log saved to {log_path}")
    print(f"Best Val Acc: {best_metric:.2f}% (Epoch {best_epoch})")
    print(f"Test Acc: {real_test_acc:.2f}%, F1: {test_f1:.2f}%")


if __name__ == "__main__":
    main()
