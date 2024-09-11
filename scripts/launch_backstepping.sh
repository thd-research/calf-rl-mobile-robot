if [[ $1 = "--ros" ]] || [[ $1 = "-r" ]]
   then
      python3.10 run.py \
               simulator=ros \
               policy=3wrobot_dyn_min_grad_clf \
               initial_conditions=3wrobot_dyn \
               system=3wrobot_dyn \
               common.sampling_time=0.02 --jobs=12
   else
      python3.10 run.py \
               policy=3wrobot_dyn_min_grad_clf \
               initial_conditions=3wrobot_dyn \
               system=3wrobot_dyn \
               common.sampling_time=0.01 \
               policy.gain=2 \
               --interactive --fps=100 
fi