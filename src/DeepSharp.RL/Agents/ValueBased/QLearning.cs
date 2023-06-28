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

        public float Epsilon { protected set; get; }
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
            var action = QTable.GetArgMax(state);
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

            var oldValue = QTable.GetValue(state, action);
            var actNext = QTable.GetArgMax(state);
            var qNext = actNext == null ? 0 : QTable.GetValue(stateNew, actNext.Value!);
            var newValue = reward + Gamma * qNext;
            var finalValue = (1 - Alpha) * oldValue + Alpha * newValue;
            QTable.Update(state, action, finalValue);
        }

        public override float Learn(int count)
        {
            var episodes = RunEpisode(1, PlayMode.EpsilonGreedy, Update);
            return episodes.Length;
        }


        public Step SampleEnv()
        {
            var action = GetEpsilonAct(Environ.Observation.Value, Epsilon);
            var step = Environ.Step(action, 0);
            if (Environ.IsComplete(0)) Environ.Reset();
            return step;
        }

        public override Episode RunEpisode(PlayMode playMode = PlayMode.Agent, Action<Step>? updateAgent = null)
        {
            return base.RunEpisode(playMode, Update);
        }
    }
}