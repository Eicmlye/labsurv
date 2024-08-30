class DeleteUninstalledCameraError(ValueError):
    def __init__(self, *args):
        super().__init__(*args)


class InstallAtExistingCameraError(ValueError):
    def __init__(self, *args):
        super().__init__(*args)
