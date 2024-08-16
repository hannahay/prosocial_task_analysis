# prosocial_task_analysis
project analyzing the behavior and the calcium imaging data recorded during pro-social paradigm


   - on day 0: actor mice receive reward pellet from both the right and the left nose port. Reward 
     delivered one second after poking. (no recipient in the cage). At the end of the session, the less 
     preferred nose port is defined as the empathic port
     ![Copy of Untitled](https://github.com/hannahay/test/assets/64631934/141bc90d-e668-425c-a96e-096a794c36cc)

   - from day 1 to day 4: recipient is in the cage, a transparent divider separates the actor and the 
     recipient. Actor mice can choose to poke to an inactive port (no reward) and to an empathic port 
     (reward delivered to recipient)
     

     ![Untitled (2)](https://github.com/hannahay/test/assets/64631934/ebb1061f-e2d2-4a10-862b-17d94a162189)

2) Data file for each session (ex: mouse1 and day0):

   one file including the neurons activity (matrix: neurons x time), the SLEAP tracking data (5 body 
   parts) and the times of poking

Analysis steps:
1) plot preference score for across days
1) find responsive neurons to poking
2) plot neurons' responses (psth) during poking and selectivity 
3) build a classifier based on neuronal activity to classify between empathic and selfish pokes
- build a classifier based on neuronal activity to classify between empathic and selfish pokes (behavioral_analysis.py)

