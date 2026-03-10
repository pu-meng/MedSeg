import warnings

def setup_warnings():
    # 1) 屏蔽 cuda.cudart 的 FutureWarning(不影响训练)
    warnings.filterwarnings(
        "ignore",
        message=r".*cuda\.cudart module is deprecated.*",
        category=FutureWarning,
    )

    # 2) 屏蔽 MONAI Orientationd labels 默认值变更提示
    warnings.filterwarnings(
        "ignore",
        message=r".*Orientationd\.__init__\:labels.*Default value changed.*",
        category=FutureWarning,
    )

    # (可选)如果你还想更“干净”,把 MONAI 的 FutureWarning 都压掉:
    # warnings.filterwarnings("ignore", category=FutureWarning, module=r"monai\..*")

    #FutureWarning:未来版本会改
    #DeprecationWarning:即将废弃
    #UserWarning:用户警告
    #r是正则表达式,raw string,字符串前加r,表示字符串不转义