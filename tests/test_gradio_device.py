from scripts.gradio_app import device_options


def test_device_options_offer_gpu_and_cpu_when_cuda_is_available() -> None:
    assert device_options(force_cpu=False, cuda_available=True) == (["GPU", "CPU"], "GPU")


def test_device_options_fall_back_to_cpu() -> None:
    assert device_options(force_cpu=False, cuda_available=False) == (["CPU"], "CPU")
    assert device_options(force_cpu=True, cuda_available=True) == (["CPU"], "CPU")
