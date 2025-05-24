import logging

log = logging.getLogger()


def check_performance_against_requirements(
    metrics, required_performance_dict, metrics_availability_checked, available_metrics
):
    """
    Check if the computed metrics meet the required performance thresholds.

    Args:
        metrics (dict): The computed metrics from the current evaluation
        required_performance_dict (dict): Dictionary of required metric thresholds
        metrics_availability_checked (bool): Whether metrics availability has been checked
        available_metrics (dict): Previously identified available metrics and their thresholds

    Returns:
        tuple: (is_performance_reached, metrics_availability_checked, available_metrics)
    """
    is_performance_reached = False

    if required_performance_dict is not None:
        # Only check which metrics are available on the first iteration with valid metrics
        if not metrics_availability_checked:
            # Determine which required metrics are available in computed metrics
            available_metrics = {
                metric: threshold
                for metric, threshold in required_performance_dict.items()
                if metric in metrics
            }
            missing_metrics = [
                metric for metric in required_performance_dict if metric not in metrics
            ]

            if missing_metrics:
                log.warning(
                    f"Required metrics {missing_metrics} not found in computed metrics. "
                    f"These metrics will be skipped when evaluating required performance."
                )

            if not available_metrics:
                log.warning(
                    "None of the required metrics are available. Cannot evaluate required performance."
                )

            metrics_availability_checked = True

        # Check if required performance is reached based on available metrics
        if available_metrics:
            # Make sure all expected metrics are still present in current iteration
            all_metrics_present = all(metric in metrics for metric in available_metrics)

            if not all_metrics_present:
                missing_now = [
                    metric for metric in available_metrics if metric not in metrics
                ]
                log.warning(
                    f"Previously available metrics {missing_now} are no longer in the computed metrics. "
                    f"Skipping performance check for these metrics on this iteration."
                )
                is_performance_reached = all(
                    metrics[metric] >= threshold
                    for metric, threshold in available_metrics.items()
                    if metric in metrics
                )
            else:
                is_performance_reached = all(
                    metrics[metric] >= threshold
                    for metric, threshold in available_metrics.items()
                )
                log.info(
                    f"Performance check on available metrics: {is_performance_reached}"
                )
        else:
            is_performance_reached = False

    return is_performance_reached, metrics_availability_checked, available_metrics
