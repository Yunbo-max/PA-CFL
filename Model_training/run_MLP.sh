#!/bin/bash
# @Author: Yunbo
# @Date:   2024-01-24 10:28:47
# @Last Modified by:   Yunbo
# @Last Modified time: 2024-02-13 14:17:51
#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/

# Start the server
python server_CNN.py &
sleep 3  # Sleep for 3s to give the server enough time to start

# Define the sheet names
# sheet_names=('0' '1' '2' '3' '5' '6' '7' '9' '10' '12' '14' '16' '17' '22')
sheet_names=('0' '1')

# Loop over the sheet names and start the clients
for i in "${sheet_names[@]}"; do
    echo "Starting client $i"
    python "client_MLP$i.py" --partition="$i" > "client_$i.log" 2>&1 &
done

# Enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
