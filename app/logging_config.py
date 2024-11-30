import logging
import sys

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    print("logging imported")
    
if __name__ == "__main__":
    setup_logging()
    print("Logging configured")
    logging.info("This is an info message")