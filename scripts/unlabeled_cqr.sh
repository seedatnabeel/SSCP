labeledprop=0.4
RUNS=5
EPOCHS=500


python ../src/unlabeled_sscp_cqr.py --dataset=concrete --runs $RUNS --epochs=$EPOCHS --labeled-prop 0.1 
python ../src/unlabeled_sscp_cqr.py --dataset=star --runs $RUNS --epochs=$EPOCHS --labeled-prop 0.1 
python ../src/unlabeled_sscp_cqr.py --dataset=community --runs $RUNS --epochs=$EPOCHS --labeled-prop 0.1 
python ../src/unlabeled_sscp_cqr.py --dataset=bike --runs $RUNS --epochs=$EPOCHS --labeled-prop 0.1 


python ../src/unlabeled_sscp_cqr.py --dataset=concrete --runs $RUNS --epochs=$EPOCHS --labeled-prop 0.2  
python ../src/unlabeled_sscp_cqr.py --dataset=star --runs $RUNS --epochs=$EPOCHS --labeled-prop 0.2 
python ../src/unlabeled_sscp_cqr.py --dataset=community --runs $RUNS --epochs=$EPOCHS --labeled-prop 0.2 
python ../src/unlabeled_sscp_cqr.py --dataset=bike --runs $RUNS --epochs=$EPOCHS --labeled-prop 0.2 


python ../src/unlabeled_sscp_cqr.py --dataset=concrete --runs $RUNS --epochs=$EPOCHS --labeled-prop 0.3 
python ../src/unlabeled_sscp_cqr.py --dataset=star --runs $RUNS --epochs=$EPOCHS --labeled-prop 0.3 
python ../src/unlabeled_sscp_cqr.py --dataset=community --runs $RUNS --epochs=$EPOCHS --labeled-prop 0.3 
python ../src/unlabeled_sscp_cqr.py --dataset=bike --runs $RUNS --epochs=$EPOCHS --labeled-prop 0.3 


python ../src/unlabeled_sscp_cqr.py --dataset=concrete --runs $RUNS --epochs=$EPOCHS --labeled-prop 0.4 
python ../src/unlabeled_sscp_cqr.py --dataset=star --runs $RUNS --epochs=$EPOCHS --labeled-prop 0.4 
python ../src/unlabeled_sscp_cqr.py --dataset=community --runs $RUNS --epochs=$EPOCHS --labeled-prop 0.4 
python ../src/unlabeled_sscp_cqr.py --dataset=bike --runs $RUNS --epochs=$EPOCHS --labeled-prop 0.4 


python ../src/unlabeled_sscp_cqr.py --dataset=concrete --runs $RUNS --epochs=$EPOCHS --labeled-prop 0.5 
python ../src/unlabeled_sscp_cqr.py --dataset=star --runs $RUNS --epochs=$EPOCHS --labeled-prop 0.5 
python ../src/unlabeled_sscp_cqr.py --dataset=community --runs $RUNS --epochs=$EPOCHS --labeled-prop 0.5 
python ../src/unlabeled_sscp_cqr.py --dataset=bike --runs $RUNS --epochs=$EPOCHS --labeled-prop 0.5 



python ../src/unlabeled_sscp_cqr.py --dataset=facebook_2 --runs $RUNS --epochs=$EPOCHS --labeled-prop 0.1  
python ../src/unlabeled_sscp_cqr.py --dataset=blog_data --runs $RUNS --epochs=$EPOCHS --labeled-prop 0.1 

python ../src/unlabeled_sscp_cqr.py --dataset=facebook_2 --runs $RUNS --epochs=$EPOCHS --labeled-prop 0.2 
python ../src/unlabeled_sscp_cqr.py --dataset=blog_data --runs $RUNS --epochs=$EPOCHS --labeled-prop 0.2 

python ../src/unlabeled_sscp_cqr.py --dataset=facebook_2 --runs $RUNS --epochs=$EPOCHS --labeled-prop 0.3  
python ../src/unlabeled_sscp_cqr.py --dataset=blog_data --runs $RUNS --epochs=$EPOCHS --labeled-prop 0.3 

python ../src/unlabeled_sscp_cqr.py --dataset=facebook_2 --runs $RUNS --epochs=$EPOCHS --labeled-prop 0.4 
python ../src/unlabeled_sscp_cqr.py --dataset=blog_data --runs $RUNS --epochs=$EPOCHS --labeled-prop 0.4 

python ../src/unlabeled_sscp_cqr.py --dataset=facebook_2 --runs $RUNS --epochs=$EPOCHS --labeled-prop 0.5 
python ../src/unlabeled_sscp_cqr.py --dataset=blog_data --runs $RUNS --epochs=$EPOCHS --labeled-prop 0.5 




