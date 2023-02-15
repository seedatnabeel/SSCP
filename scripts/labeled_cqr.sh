labeledprop=0.4
RUNS=5
EPOCHS=500

echo "CQR"

python ../src/labeled_sscp_cqr.py --dataset=concrete --runs $RUNS --epochs=$EPOCHS --labeled-prop $labeledprop

python ../src/labeled_sscp_cqr.py --dataset=star --runs $RUNS --epochs=$EPOCHS --labeled-prop $labeledprop 

python ../src/labeled_sscp_cqr.py --dataset=community --runs $RUNS --epochs=$EPOCHS --labeled-prop $labeledprop   

python ../src/labeled_sscp_cqr.py --dataset=bike --runs $RUNS --epochs=$EPOCHS --labeled-prop $labeledprop 

python ../src/labeled_sscp_cqr.py --dataset=facebook_1 --runs $RUNS --epochs=$EPOCHS --labeled-prop $labeledprop 

python ../src/labeled_sscp_cqr.py --dataset=facebook_2 --runs $RUNS --epochs=$EPOCHS --labeled-prop $labeledprop 

python ../src/labeled_sscp_cqr.py --dataset=blog_data --runs $RUNS --epochs=$EPOCHS --labeled-prop $labeledprop 


