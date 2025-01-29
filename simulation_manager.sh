#!/usr/bin/bash

# Train original model
#for (( i=1; i <= 500; ++i ))
#do
#  echo "$i"
#  python -m medical_federated_unlearning.main_fl_2
#done

# Train retrained model
#for (( j=0; j <= 5; ++j ))
#do
#  for (( i=1; i <= 200; ++i ))
#  do
#    echo "$i"
#    python -m medical_federated_unlearning.main_fl_2 retraining=True unlearned_cid=[$j]
#  done
#done

# Try natural forgetting
#for (( j=0; j <= 5; ++j ))
#do
#  for (( i=1; i <= 100; ++i ))
#  do
#    python -m medical_federated_unlearning.main_fl_2 resume_training=True resuming_after_unlearning.algorithm="natural" unlearned_cid=[$j]
#  done
#done

# unlearning
#for (( j=0; j <= 4; ++j ))
#do
#  for (( i=1; i <= 15; ++i ))
#  do
#    python -m medical_federated_unlearning.main_fl_unlearn resume_training=True resuming_after_unlearning.algorithm="pseudo_gradient_ascent_single" unlearned_cid=[$j] resuming_after_unlearning.unlearning_lr=7.0 resuming_after_unlearning.unlearning_epochs=1
#  done
#done
#
#for (( j=0; j <= 4; ++j ))
#do
#  for (( i=1; i <= 20; ++i ))
#  do
#    python -m medical_federated_unlearning.main_fl_unlearn resume_training=True resuming_after_unlearning.algorithm="pseudo_gradient_ascent_single" unlearned_cid=[$j] resuming_after_unlearning.unlearning_lr=8.0 resuming_after_unlearning.unlearning_epochs=1
#  done
#done

#for (( j=2; j <= 3; ++j ))
#do
#  for (( i=1; i <= 100; ++i ))
#  do
#    python -m medical_federated_unlearning.main_fl_unlearn resume_training=True resuming_after_unlearning.algorithm="pseudo_gradient_ascent_single" unlearned_cid=[$j] resuming_after_unlearning.unlearning_lr=19.0 resuming_after_unlearning.unlearning_epochs=1
#  done
#done

for (( j=3; j <= 3; ++j ))
do
  for (( i=1; i <= 150; ++i ))
  do
    python -m medical_federated_unlearning.main_fl_unlearn resume_training=True resuming_after_unlearning.algorithm="pseudo_gradient_ascent" unlearned_cid=[$j] resuming_after_unlearning.unlearning_lr=120.0 resuming_after_unlearning.unlearning_epochs=1
  done
done

for (( j=3; j <= 3; ++j ))
do
  for (( i=1; i <= 150; ++i ))
  do
    python -m medical_federated_unlearning.main_fl_unlearn resume_training=True resuming_after_unlearning.algorithm="pseudo_gradient_ascent" unlearned_cid=[$j] resuming_after_unlearning.unlearning_lr=40.0 resuming_after_unlearning.unlearning_epochs=1
  done
done

#for (( j=0; j <= 4; ++j ))
#do
#  for (( i=1; i <= 60; ++i ))
#  do
#    python -m medical_federated_unlearning.main_fl_unlearn resume_training=True resuming_after_unlearning.algorithm="pseudo_gradient_ascent_single" unlearned_cid=[$j] resuming_after_unlearning.unlearning_lr=30.0 resuming_after_unlearning.unlearning_epochs=1
#  done
#done