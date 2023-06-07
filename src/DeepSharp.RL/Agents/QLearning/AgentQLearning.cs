﻿using DeepSharp.RL.Models;
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
            Values = new Dictionary<Observation, float>();
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
        public Dictionary<Observation, float> Values { set; get; }

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
                var reward = getReward(new RewardKey(trasitKey.State, trasitKey.Action, i.Key));
                var value = reward.Value + Environ.Gamma * getValue(i.Key);
                activaValue += 1f * i.Value / total * value;
            }

            return activaValue;
        }

        private Dictionary<Observation, int> getTransit(TrasitKey traitKey)
        {
            var key = Transits.Keys.First(a => a.Action.Value.Equals(traitKey.Action.Value) &&
                                               a.State.Value.Equals(traitKey.State.Value));
            return Transits[key];
        }

        private Reward getReward(RewardKey rewardKey)
        {
            var key = Rewards.Keys.First(a => a.Action.Value.Equals(rewardKey.Action.Value) &&
                                              a.State.Value.Equals(rewardKey.State.Value) &&
                                              a.NewState.Value.Equals(rewardKey.NewState.Value));
            return Rewards[key];
        }

        private float getValue(Observation observation)
        {
            var key = Values.Keys.FirstOrDefault(a => a.Value.Equals(observation.Value));
            if (key != null) return Values[key];

            Values[observation] = 0;
            return Values[observation];
        }

        public void ValueIteration()
        {
            var stateList = Rewards
                .Select(a => a.Key.State)
                .Distinct();
            var actionList = Enumerable.Range(0, ActionSize)
                .Select(a => new Action(torch.from_array(new long[] {a})))
                .ToList();
            foreach (var state in stateList)
            {
                var maxStateValue = actionList
                    .Select(a => GetActionValue(new TrasitKey(state, a)))
                    .Max();

                Values[state] = maxStateValue;
            }
        }


        private Action SelectAction(Observation state)
        {
            var actionSpace = Enumerable.Range(1, ActionSize)
                .Select(a => new Action(torch.from_array(new[] {a})))
                .ToList();

            var value = actionSpace.MaxBy(a => GetActionValue(new TrasitKey(state, a)));

            return value!;
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