"""支持 python -m skinscan，与安装生成的 skinscan 命令共用入口与退出码。"""
from .cli import main

raise SystemExit(main())
