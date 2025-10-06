#!/usr/bin/env python3
"""
Web application launcher for Van Gogh Image Generator
Provides a simple way to start the web interface
"""
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from image_generator import VanGoghWebInterface
from config import web_config

def main():
    """Launch the web application"""
    print("🎨 Van Gogh Image Generator")
    print("=" * 50)
    print("Starting web interface...")
    
    # Create and launch the interface
    interface = VanGoghWebInterface()
    
    print(f"Web interface will be available at: http://{web_config.host}:{web_config.port}")
    print("Press Ctrl+C to stop the server")
    
    try:
        interface.launch(
            server_name=web_config.host,
            server_port=web_config.port,
            share=web_config.share,
            debug=web_config.debug
        )
    except KeyboardInterrupt:
        print("\nShutting down web interface...")
    except Exception as e:
        print(f"Error launching web interface: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
