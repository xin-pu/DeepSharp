using DeepSharp.RL.Agents;
using DeepSharp.RL.Environs;

namespace DeepSharp.RL.Trainers
{
    public class RLTrainer
    {
        private TrainerCallBack? callback;

        public RLTrainer(Agent agent)
        {
            Agent = agent;
        }

        public RLTrainer(Agent agent, Action<object> print)
        {
            Agent = agent;
            Print = print;
        }

        public Agent Agent { set; get; }

        public TrainerCallBack? Callback
        {
            set
            {
                callback = value;
                if (callback != null)
                    callback.RlTrainer = this;
            }
            get => callback;
        }

        public Action<object>? Print { set; get; }


        public virtual void Train(
            float preReward,
            int trainEpoch,
            string saveFolder = "",
            int testEpisodes = -1,
            int testInterval = 5,
            bool autoSave = false)
        {
            OnTrainStart();

            var valEpoch = 0;
            foreach (var epoch in Enumerable.Range(1, trainEpoch))
            {
                OnLearnStart(epoch);
                var outcome = Agent.Learn();
                OnLearnEnd(epoch, outcome);


                if (testEpisodes <= 0)
                    continue;

                if (epoch % testInterval == 0)
                {
                    valEpoch++;
                    OnValStart(valEpoch);
                    var episodes = Agent.RunEpisodes(testEpisodes);
                    OnValStop(valEpoch, episodes);

                    var valReward = episodes.Average(e => e.SumReward.Value);

                    if (valReward < preReward)
                        continue;

                    /// val reward > pre reward
                    /// save and break from training
                    if (autoSave)
                    {
                        OnSaveStart();
                        Agent.Save(Path.Combine(saveFolder, $"[{Agent}]_{epoch}_{valReward:F2}.st"));
                        OnSaveEnd();
                    }

                    break;
                }
            }

            OnTrainEnd();
        }


        public virtual void Train(RLTrainOption tp)
        {
            Train(tp.StopReward, tp.TrainEpoch, tp.SaveFolder, tp.ValEpisode, tp.ValInterval);
        }

        public virtual void Val(int valEpoch)
        {
            var episodes = Agent.RunEpisodes(valEpoch);
            var aveReward = episodes.Average(a => a.SumReward.Value);
            Print?.Invoke($"[Val] {valEpoch:D5}\tR:[{aveReward}]");
            foreach (var episode in episodes) Print?.Invoke(episode);
        }




        #region MyRegion

        protected virtual void OnTrainStart()
        {
            Print?.Invoke($"[{Agent}] start training.");
            Callback?.OnTrainStart();
        }

        protected virtual void OnTrainEnd()
        {
            Print?.Invoke($"[{Agent}] stop training.");
            Callback?.OnTrainEnd();
        }


        protected virtual void OnLearnStart(int epoch)
        {
            Callback?.OnLearnStart(epoch);
        }


        protected virtual void OnLearnEnd(int epoch, LearnOutcome outcome)
        {
            Print?.Invoke($"[Tra]\t{epoch:D5}\t{outcome}");
            Callback?.OnLearnEnd(epoch, outcome);
        }

        protected virtual void OnValStart(int epoch)
        {
            Callback?.OnValStart(epoch);
        }

        protected virtual void OnValStop(int epoch, Episode[] episodes)
        {
            var aveReward = episodes.Average(a => a.SumReward.Value);
            Print?.Invoke($"[Val]\t{epoch:D5}\tE:{episodes.Length}:\tR:{aveReward:F4}");
            Callback?.OnValEnd(epoch, episodes);
        }

        protected virtual void OnSaveStart()
        {
            Callback?.OnSaveStart();
        }

        protected virtual void OnSaveEnd()
        {
            Callback?.OnSaveEnd();
        }

        #endregion
    }
}