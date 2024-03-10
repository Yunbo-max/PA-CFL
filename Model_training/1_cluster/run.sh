#!/bin/bash
# @Author: Yunbo
# @Date:   2024-02-13 14:13:17
# @Last Modified by:   Yunbo
# @Last Modified time: 2024-03-09 12:25:50
#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/

# Parse command-line arguments
CONFIG_FILE="config.json"  # Default configuration file path
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -c|--config)
        CONFIG_FILE="$2"
        shift # past argument
        shift # past value
        ;;
        *)    # unknown option
        echo "Unknown option: $1"
        exit 1
        ;;
    esac
done

# Start the server
python server_CNN.py &
sleep 3  # Sleep for 3s to give the server enough time to start

# Load configuration from JSON file
config=$(cat "$CONFIG_FILE")

# Define the sheet names
# sheet_names=('0' '1' '2' '3' '5' '6' '7' '9' '10' '12' '14' '16' '17' '22')
sheet_names=('Southeast Asia' 'South Asia' 'Oceania' 'Eastern Asia' 'West of USA' 'US Center' 'West Africa' 'North Africa' 'Western Europe' 'Central America' 'South America' 'Southern Europe' 'East of USA' 'South of  USA')




# Loop over the sheet names and start the clients
for i in "${sheet_names[@]}"; do
    echo "Starting client $i"
    python client.py --client_number="$i" --config_file="$CONFIG_FILE" > "client_$i.log" 2>&1 &
done

# Enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
