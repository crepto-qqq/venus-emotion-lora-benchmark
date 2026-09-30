#!/usr/bin/env python3
import sys
from run_lora_training import main
if __name__=="__main__":
    if "--mode" not in sys.argv: sys.argv[1:1]=["--mode","pilot"]
    main()
