using DeepSharp.RL.Enumerates;
using DeepSharp.RL.Environs;
using FluentAssertions;

namespace DeepSharp.RL.Agents
{
    /// <summary>
    ///     This is a demo of ValueIteration
    ///     [Deep Reinforcement Learning Hands-On Second Edition](Russia,Maxim Lapan)
    ///     [Chap5.5]
    /// </summary>
    public class ValueIteration : Agent

    {
        public ValueIteration(Environ<Space, Space> env, int t, float gamma = 0.9f)
            : base(env, "ValueIteration")
        {
            Rewards = new Dictionary<RewardKey, Reward>();
            Transits = new Dictionary<TransitKey, Dictionary<torch.Tensor, int>>();
            Values = new Dictionary<torch.Tensor, float>();
            T = t;
            Gamma = gamma;
        }

        public int T { protected set; get; }
        public float Gamma { protected set; get; }

        /// <summary>
        ///     奖励表
        /// </summary>
        public Dictionary<RewardKey, Reward> Rewards { set; get; }

        /// <summary>
        ///     转移表
        /// </summary>
        public Dictionary<TransitKey, Dictionary<torch.Tensor, int>> Transits { set; get; }

        /// <summary>
        ///     价值表
        /// </summary>
        public Dictionary<torch.Tensor, float> Values { set; get; }


        /// <summary>
        ///     Select  Action According with Latest Observation
        ///     选择动作
        /// </summary>
        /// <param name="state"></param>
        /// <returns></returns>
        public override Act GetPolicyAct(torch.Tensor state)
        {
            Rewards.Count.Should().BeGreaterThan(0, "Rewards Table is Empty, You should learn first.");
            Transits.Count.Should().BeGreaterThan(0, "Transits Table is Empty, You should learn first.");
            Values.Count.Should().BeGreaterThan(0, "Values Table is Empty, You should learn first.");

            /// Step 1 Get Action Space According Current State from Transits
            var actionSpace = Transits.Keys
                .Where(a => a.State.Equals(state))
                .ToList();

            var valueDict = actionSpace
                .ToDictionary(a => a.Act, a => GetActionValue(a));
            var maxValue = valueDict.Values.Max();
            var maxActs = valueDict
                .Where(a => Math.Abs(a.Value - maxValue) < 1E-4)
                .Select(a => a.Key)
                .ToList();

            if (maxActs.Count == 1) return new Act(maxActs.First());

            var probs = Enumerable.Repeat(1f, maxActs.Count).ToArray();
            var actIndex = torch.multinomial(torch.tensor(probs), 1, true).ToInt32();
            return new Act(maxActs[actIndex]);
        }


        public override LearnOutcome Learn()
        {
            var episodes = RunEpisodes(T, PlayMode.Sample);
            UpdateValueIteration();
            return new LearnOutcome(episodes);
        }

        public override void Save(string path)
        {
            throw new NotImplementedException();
        }

        public override void Load(string path)
        {
            throw new NotImplementedException();
        }

        public void Update(Episode episode)
        {
            episode.Steps.ForEach(UpdateTables);
        }


        /// <summary>
        ///     Update Value Iteration
        ///     更新价值表
        /// </summary>
        public void UpdateValueIteration()
        {
            var stateList = Transits.Select(a => a.Key.State).Distinct();

            foreach (var state in stateList)
            {
                var actionList = Transits.Keys
                    .Where(a => a.State.Equals(state))
                    .Select(a => a.Act);

                var maxStateValue = actionList
                    .Select(a => GetActionValue(new TransitKey(state, a)))
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


            var transitsKey = new TransitKey(state, act);
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
                .Where(a => a.Equals(newState.Value!))
                .ToList();
            if (newStateKeys.Any())
                sonDict[newStateKeys.First()]++;
            else
                sonDict[newTensor] = 1;
        }

        private void UpdateTables(Step step)
        {
            UpdateTables(step.PreState, step.Action, step.PostState, step.Reward);
        }

        /// <summary>
        ///     状态和动作的近似价值Q(s,a) = 每个状态的概率乘以状态价值
        ///     根据Bellman方程，它也等于立即奖励和折扣长期状态价值之和
        /// </summary>
        /// <param name="transitKey"></param>
        /// <returns></returns>
        private float GetActionValue(TransitKey transitKey)
        {
            var targetCounts = getTransit(transitKey);
            var total = targetCounts.Sum(a => a.Value);
            var activaValue = 0f;
            foreach (var i in targetCounts)
            {
                var reward = getReward(new RewardKey(transitKey.State, transitKey.Act, i.Key));
                var value = reward.Value + Gamma * getValue(i.Key);
                activaValue += 1f * i.Value / total * value;
            }

            return activaValue;
        }

        private Dictionary<torch.Tensor, int> getTransit(TransitKey traitKey)
        {
            try
            {
                var key = Transits.Keys
                    .First(a => a.Act.Equals(traitKey.Act) &&
                                a.State.Equals(traitKey.State));

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
                    .First(a => a.Act.Equals(rewardKey.Act) &&
                                a.State.Equals(rewardKey.State) &&
                                a.NewState.Equals(rewardKey.NewState));
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
                var key = Values.Keys.First(a => a.Equals(observation));
                return Values[key];
            }
            catch
            {
                return 0;
            }
        }
    }
}