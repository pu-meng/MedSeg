import warnings
"""
r"",r=raw
"\n"表示换行
r"\n"这里的\n,就是单纯的符号\n
f"",f=format(格式化字符串)
f的作用是字符串里嵌入变量

先进行python解释,类似r""这里的r保证里面的原模原样;
然后进行正则表达式解释,
"""

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

  