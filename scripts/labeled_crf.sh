echo "RUNNING CONCRETE FOR 5 RANDOM RUNS..."

python ../src/labeled_sscp.py --dataset=concrete --runs 5 --epochs=500 --labeled-prop 0.4 

echo "RUNNING STAR FOR 5 RANDOM RUNS..."

python ../src/labeled_sscp.py --dataset=star --runs 5 --epochs=500 --labeled-prop 0.4

echo "RUNNING COMMUNITY FOR 5 RANDOM RUNS..."

python ../src/labeled_sscp.py --dataset=community --runs 5 --epochs=500 --labeled-prop 0.4

echo "RUNNING BIKE FOR 5 RANDOM RUNS..."

python ../src/labeled_sscp.py --dataset=bike --runs 5 --epochs=500 --labeled-prop 0.4 

echo "RUNNING BLOG FOR 5 RANDOM RUNS..."

python ../src/labeled_sscp.py --dataset=blog_data --runs 5 --epochs=100 --labeled-prop 0.4 

echo "RUNNING FB FOR 5 RANDOM RUNS..."

python ../src/labeled_sscp.py --dataset=facebook_2 --runs 5 --epochs=500 --labeled-prop 0.4

