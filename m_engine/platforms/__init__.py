import logging

logger = logging.getLogger(__name__)


def cuda_platform_plugin():
    is_cuda = False
    logger.debug("Checking if CUDA platform is available.")
    try:
        from vllm.utils.import_utils import import_pynvml

        pynvml = import_pynvml()
        pynvml.nvmlInit()
        try:
            # NOTE: Edge case: vllm cpu build on a GPU machine.
            # Third-party pynvml can be imported in cpu build,
            # we need to check if vllm is built with cpu too.
            # Otherwise, vllm will always activate cuda plugin
            # on a GPU machine, even if in a cpu build.
            is_cuda = (
                pynvml.nvmlDeviceGetCount() > 0
                and not vllm_version_matches_substr("cpu")
            )
            if pynvml.nvmlDeviceGetCount() <= 0:
                logger.debug(
                    "CUDA platform is not available because no GPU is found.")
            if vllm_version_matches_substr("cpu"):
                logger.debug(
                    "CUDA platform is not available because vLLM is built with CPU."
                )
            if is_cuda:
                logger.debug("Confirmed CUDA platform is available.")
        finally:
            pynvml.nvmlShutdown()
    except Exception as e:
        logger.debug(
            "Exception happens when checking CUDA platform: %s", str(e))
        if "nvml" not in e.__class__.__name__.lower():
            # If the error is not related to NVML, re-raise it.
            raise e

        # CUDA is supported on Jetson, but NVML may not be.
        import os

        def cuda_is_jetson() -> bool:
            return os.path.isfile("/etc/nv_tegra_release") or os.path.exists(
                "/sys/class/tegra-firmware"
            )

        if cuda_is_jetson():
            logger.debug("Confirmed CUDA platform is available on Jetson.")
            is_cuda = True
        else:
            logger.debug("CUDA platform is not available because: %s", str(e))

    return "vllm.platforms.cuda.CudaPlatform" if is_cuda else None
