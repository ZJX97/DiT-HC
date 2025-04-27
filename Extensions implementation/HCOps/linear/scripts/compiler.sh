CORE=10

for pid in $(pgrep clang++); do
  taskset -pc $CORE $pid
  CORE=$((CORE+1))
done 
