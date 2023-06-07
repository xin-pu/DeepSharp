using DeepSharp.RL.Models;
using Action = DeepSharp.RL.Models.Action;

namespace DeepSharp.RL.Agents
{
    public class AgentQLearning : Agent
    {
        public AgentQLearning(Environ env)
            : base(env)
        {
            Rewards = new Dictionary<RewardKey, Reward>();
            Transits = new Dictionary<TrasitKey, Dictionary<Observation, int>>();
            Values = new Dictionary<TrasitKey, float>();
        }

        /// <summary>
        ///     奖励表
        /// </summary>
        public Dictionary<RewardKey, Reward> Rewards { set; get; }

        /// <summary>
        ///     转移表
        /// </summary>
        public Dictionary<TrasitKey, Dictionary<Observation, int>> Transits { set; get; }

        /// <summary>
        ///     价值表
        /// </summary>
        public Dictionary<TrasitKey, float> Values { set; get; }

        public override Action PredictAction(Observation reward)
        {
            throw new NotImplementedException();
        }

        public override float Learn(Episode[] steps)
        {
            throw new NotImplementedException();
        }


        public void RunRandom(Environ environ, int count)
        {
            foreach (var i in Enumerable.Range(0, count))
            {
                var observation = environ.Observation;
                var action = environ.Sample();
                var newObservation = environ.UpdateEnviron(action);
                var reward = environ.GetReward(newObservation);

                UpdateTables(observation!, action, newObservation, reward);
                environ.Observation = (Observation) newObservation.Clone();

                if (environ.StopEpoch(i))
                    environ.Reset();
            }
        }

        private void UpdateTables(Observation state, Action action, Observation newState, Reward reward)
        {
            ///Step 1 更新奖励表
            var rewardKey = new RewardKey(state, action, newState);
            var existRewardKey = Rewards.Keys.Where(a =>
                    a.Action.Value!.Equals(action.Value!) &&
                    a.State.Value!.Equals(state.Value!) &&
                    a.NewState.Value!.Equals(newState.Value!))
                .ToList();

            var finalRewardKey = existRewardKey.Any() ? existRewardKey.First() : rewardKey;
            Rewards[finalRewardKey] = reward;


            var transitsKey = new TrasitKey(state, action);
            var existTransitKey = Transits.Keys.Where(a =>
                    a.Action.Value!.Equals(action.Value!) &&
                    a.State.Value!.Equals(state.Value!))
                .ToList();

            /// Step 2 更新转移表
            Dictionary<Observation, int> sonDict;
            if (existTransitKey.Any())
            {
                sonDict = Transits[existTransitKey.First()];
            }
            else
            {
                sonDict = new Dictionary<Observation, int>();
                Transits[transitsKey] = sonDict;
            }

            var newStateKeys = sonDict.Keys
                .Where(a => a.Value!.Equals(newState.Value!))
                .ToList();
            if (newStateKeys.Any())
                sonDict[newStateKeys.First()]++;
            else
                sonDict[newState] = 1;
        }
    }
}