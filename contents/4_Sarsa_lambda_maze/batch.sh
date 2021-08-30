rm *.log
python3 ./run_this.py QLearn
python3 ./run_this.py Sarsa
python3 ./run_this.py SarsaLambda
grep 'game over' *.log
