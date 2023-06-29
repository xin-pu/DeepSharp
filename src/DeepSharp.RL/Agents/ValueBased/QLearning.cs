using DeepSharp.RL.Environs;

namespace DeepSharp.RL.Agents
{
    public class QLearning : ValueAgent
    {
        public QLearning(Environ<Space, Space> env,
            float epsilon = 0.1f,
            float alpha = 0.2f,
            float gamma = 0.9f) :
            base(env)
        {
            Epsilon = epsilon;
            Alpha = alpha;
            Gamma = gamma;
        }


        public float Alpha { protected set; get; }
        public float Gamma { protected set; get; }

        /// <summary>
        ///     ε 贪心策略
        ///     利用和探索 策略
        /// </summary>
        /// <param name="state"></param>
        /// <returns></returns>
        public override Act GetPolicyAct(torch.Tensor state)
        {
            var action = ValueTable.GetBestAct(state);
            return action ?? GetSampleAct();
        }

        public override void Update(Episode episode)
        {
        }

        public void Update(Step step)
        {
            var state = step.State.Value!;
            var action = step.Action.Value!;
            var stateNew = step.StateNew.Value!;
            var reward = step.Reward.Value!;

            var currentTransit = new TransitKey(state, action);

            var bestValue = ValueTable.GetBestValue(stateNew);
            var newValue = reward + Gamma * bestValue;
            var oldValue = ValueTable[currentTransit];
            var finalValue = oldValue * (1 - Alpha) + newValue * Alpha;

            ValueTable[currentTransit] = finalValue;
        }

        public override float Learn(int count)
        {
            var episodes = RunEpisode(1, PlayMode.EpsilonGreedy, Update);
            return episodes.Length;
        }


        public Step SampleEnv()
        {
            var action = GetEpsilonAct(Environ.Observation!.Value!);
            var step = Environ.Step(action, 0);
            if (Environ.IsComplete(0)) Environ.Reset();
            return step;
        }

        public override Episode RunEpisode(PlayMode playMode = PlayMode.Agent, Action<Step>? updateAgent = null)
        {
            return base.RunEpisode(playMode, Update);
        }

        public override string ToString()
        {
            return "QLearning";
        }
    }
}