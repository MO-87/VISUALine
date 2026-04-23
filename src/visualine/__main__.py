import sys
from visualine.api.server import start_server

def main():
    print("Starting VISUALine AI Suite Backend...")
    
    ## use reload=False for production/packaged apps
    reload_mode = "--dev" in sys.argv
    start_server(host="127.0.0.1", port=8000, reload=reload_mode)

if __name__ == "__main__":
    main()