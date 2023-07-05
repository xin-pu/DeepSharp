using DeepSharp.RL.Agents;
using DeepSharp.RL.Environs;

namespace DeepSharp.RL.Trainers
{
    public class RLTrainer
    {
        protected RLTrainer(Agent agent)
        {
            Agent = agent;
        }

        public Agent Agent { set; get; }


        public TrainerCallBack? Callback { set; get; }
        public Action<object>? Print { set; get; }

        public virtual void Train(
            float preReward,
            int trainEpoch,
            string saveFolder,
            int testEpisodes = -1,
            int testInterval = 5)
        {
            OnTrainStart();

            foreach (var epoch in Enumerable.Range(0, trainEpoch))
            {
                OnLearnStart(epoch);

                var outcome = Agent.Learn();

                OnLearnEnd(epoch, outcome);


                if (testEpisodes <= 0)
                    continue;

                if (epoch % testInterval == 0)
                {
                    OnValStart(epoch);
                    var episodes = Agent.RunEpisodes(testEpisodes);
                    OnValStop(epoch, episodes);

                    var valReward = episodes.Average(e => e.SumReward.Value);
                    if (valReward >= preReward)
                    {
                        OnSaveStart();
                        Agent.Save(Path.Combine(saveFolder, $"[{Agent}]_{epoch}_{valReward:F2}.st"));
                        OnSaveEnd();
                        break;
                    }
                }
            }
        }


        public virtual void Train(RLTrainOption tp)
        {
            Train(tp.StopReward, tp.TrainEpoch, tp.SaveFolder, tp.ValEpisode, tp.ValInterval);
        }


        public virtual void OnTrainStart()
        {
            Print?.Invoke($"[{Agent}] start training.");
            Callback?.OnTrainStart();
        }

        public virtual void OnTrainEnd()
        {
            Print?.Invoke($"[{Agent}] stop training.");
            Callback?.OnTrainEnd();
        }


        public virtual void OnLearnStart(int epoch)
        {
            Callback?.OnLearnStart(epoch);
        }


        public virtual void OnLearnEnd(int epoch, LearnOutcome outcome)
        {
            Print?.Invoke($"Learn {epoch:D5}:\t {outcome}");
            Callback?.OnLearnEnd(epoch, outcome);
        }

        public virtual void OnValStart(int epoch)
        {
            Callback?.OnValStart(epoch);
        }

        public virtual void OnValStop(int epoch, Episode[] episodes)
        {
            var aveReward = episodes.Average(a => a.SumReward.Value);
            Print?.Invoke($"[{Agent}];\tVal:[{episodes.Length}]:\tR:[{aveReward:F2}]");
            Callback?.OnValEnd(epoch, episodes);
        }

        public virtual void OnSaveStart()
        {
            Callback?.OnSaveStart();
        }

        public virtual void OnSaveEnd()
        {
            Callback?.OnSaveEnd();
        }
    }
}