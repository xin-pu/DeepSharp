using DeepSharp.RL.Environs;

namespace DeepSharp.RL.Agents
{
    public class AgentQLearning : Agent

    {
        public AgentQLearning(Environ<Space, Space> env)
            : base(env)
        {
            Rewards = new Dictionary<RewardKey, Reward>();
            Transits = new Dictionary<TrasitKey, Dictionary<torch.Tensor, int>>();
            Values = new Dictionary<torch.Tensor, float>();
        }

        /// <summary>
        ///     奖励表
        /// </summary>
        public Dictionary<RewardKey, Reward> Rewards { set; get; }

        /// <summary>
        ///     转移表
        /// </summary>
        public Dictionary<TrasitKey, Dictionary<torch.Tensor, int>> Transits { set; get; }

        /// <summary>
        ///     价值表
        /// </summary>
        public Dictionary<torch.Tensor, float> Values { set; get; }

        public override Act PredictAction(Observation state)
        {
            var actionSpace = Transits.Keys
                .Where(a => a.State.Equals(state.Value!))
                .ToList();

            var trasitKey = actionSpace.MaxBy(a => GetActionValue(a));

            return new Act(trasitKey.Act);
        }

        public override float Learn(Episode[] steps)
        {
            throw new NotImplementedException();
        }


        public void RunRandom(Environ<Space, Space> environ, int count)
        {
            foreach (var i in Enumerable.Range(0, count))
            {
                var observation = environ.Observation;
                var step = environ.SampleStep(i);

                UpdateTables(observation!, step.Action, step.Observation, step.Reward);
                environ.Observation = step.Observation;

                if (environ.IsComplete(0))
                {
                    environ.Reset();
                }
            }
        }


        public void ValueIteration()
        {
            var stateList = Transits.Select(a => a.Key.State).Distinct();

            foreach (var state in stateList)
            {
                var actionList = Transits.Keys
                    .Where(a => a.State.Equals(state))
                    .Select(a => a.Act);

                var maxStateValue = actionList
                    .Select(a => GetActionValue(new TrasitKey(state, a)))
                    .Max();

                Values[state] = maxStateValue;
            }
        }


        private void UpdateTables(Observation state, Act act, Observation newState, Reward reward)
        {
            var startTensor = state.Value!;
            var newTensor = newState.Value!;
            var action = act.Value!;

            ///Step 1 更新奖励表
            var rewardKey = new RewardKey(state, act, newState);
            var existRewardKey = Rewards.Keys.Where(a =>
                    a.Act.Equals(action) &&
                    a.State.Equals(startTensor) &&
                    a.NewState.Equals(newTensor))
                .ToList();

            var finalRewardKey = existRewardKey.Any() ? existRewardKey.First() : rewardKey;
            Rewards[finalRewardKey] = reward;


            var transitsKey = new TrasitKey(state, act);
            var existTransitKey = Transits.Keys.Where(a =>
                    a.Act.Equals(act.Value!) &&
                    a.State.Equals(state.Value!))
                .ToList();

            /// Step 2 更新转移表
            Dictionary<torch.Tensor, int> sonDict;
            if (existTransitKey.Any())
            {
                sonDict = Transits[existTransitKey.First()];
            }
            else
            {
                sonDict = new Dictionary<torch.Tensor, int>();
                Transits[transitsKey] = sonDict;
            }

            var newStateKeys = sonDict.Keys
                .Where(a => a!.Equals(newState.Value!))
                .ToList();
            if (newStateKeys.Any())
                sonDict[newStateKeys.First()]++;
            else
                sonDict[newTensor] = 1;
        }

        /// <summary>
        ///     状态和动作的近似价值Q(s,a) = 每个状态的概率乘以状态价值
        ///     根据Bellman方程，它也等于立即奖励和折扣长期状态价值之和
        /// </summary>
        /// <param name="trasitKey"></param>
        /// <returns></returns>
        private float GetActionValue(TrasitKey trasitKey)
        {
            var targetCounts = getTransit(trasitKey);
            var total = targetCounts.Sum(a => a.Value);
            var activaValue = 0f;
            foreach (var i in targetCounts)
            {
                var reward = getReward(new RewardKey(trasitKey.State, trasitKey.Act, i.Key!));
                var value = reward.Value + Environ.Gamma * getValue(i.Key);
                activaValue += 1f * i.Value / total * value;
            }

            return activaValue;
        }

        private Dictionary<torch.Tensor, int> getTransit(TrasitKey traitKey)
        {
            try
            {
                var key = Transits.Keys
                    .First(a => a.Act!.Equals(traitKey.Act!) &&
                                a.State!.Equals(traitKey.State!));

                return Transits[key];
            }
            catch (Exception)
            {
                return new Dictionary<torch.Tensor, int> {[traitKey.State] = 0};
            }
        }

        private Reward getReward(RewardKey rewardKey)
        {
            try
            {
                var key = Rewards.Keys
                    .First(a => a.Act!.Equals(rewardKey.Act!) &&
                                a.State!.Equals(rewardKey.State!) &&
                                a.NewState!.Equals(rewardKey.NewState!));
                return Rewards[key];
            }
            catch (Exception)
            {
                return new Reward(0);
            }
        }

        private float getValue(torch.Tensor observation)
        {
            try
            {
                var key = Values.Keys.First(a => a!.Equals(observation!));
                return Values[key];
            }
            catch
            {
                return 0;
            }
        }
    }
}