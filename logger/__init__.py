import json
from pathlib import Path
import datetime as dt
from typing import Optional, Dict, Union, Any, List, Literal


DEFAULT_LOGPATH = Path("../.log/log.json")

class Logger:
    def write(self, log: Union[List, Dict], 
              mode: Literal["append", "overwrite"]="append"):
        if mode in ["overwrite"]:
            with self.logpath_obj.open("w") as f:
                newlog_str = json.dumps(log)
                f.write(newlog_str)
        elif mode in ["append"]:
            newlog = self.log_obj.append(log)
            with self.logpath_obj.open("w") as f:
                newlog_str = json.dumps(newlog)
                f.write(newlog_str)
        else: raise ValueError("wrong mode entered - only accepts 'append' and 'overwrite'")

    def __init__(self,
                 params: Dict[str, Any],
                 runno: Optional[int]=None,
                 logpath: Optional[Union[Path, str]]=DEFAULT_LOGPATH,
                 **kwargs):
        if logpath is None: logpath = DEFAULT_LOGPATH
        self.logpath_obj = logpath if isinstance(logpath, Path) else Path(logpath)
        if not self.logpath_obj.exists():
            self.logpath_obj.parent.mkdir()
            self.log_obj = []
        else:
            self.log_obj = json.load(self.logpath_obj.as_posix())
        assert isinstance(self.log_obj, List), "log file msut be in list format!"
        self.runno = runno
        self.params = params
        self.newlog = {"runno": runno,
                       "performance": {},
                       **params,
                       **kwargs}
        self.log_obj.append(self.newlog)
        self.write(self.newlog, mode="append")
        
    
    def log(self, item_key: str, item_val: Any) -> None:
        if item_key in self.log_obj[-1].keys():
            self.log_obj[-1][item_key] = item_val
        else:
            self.log_obj[-1].update({item_key: item_val})
        self.write(self.log_obj, mode="overwrite")