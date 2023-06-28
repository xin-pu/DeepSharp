using DeepSharp.RL.Environs;
using MathNet.Numerics.Random;

namespace DeepSharp.RL.Agents
{
    public abstract class ValueAgent : Agent
    {
        protected ValueAgent(Environ<Space, Space> env)
            : base(env)
        {
            QTable = new QTable();
        }

        public QTable QTable { set; get; }
    }


    public class QLearning : ValueAgent
    {
        public float Epsilon { protected set; get; }
        public float Alpha { protected set; get; }
        public float Gamma { protected set; get; }

        public QLearning(Environ<Space, Space> env, float epsilon = 0.1f) : base(env)
        {
            Epsilon = epsilon;
            Alpha = 0.2f;
            Gamma = 0.9f;
        }

        /// <summary>
        ///     ε 贪心策略
        ///     利用和探索 策略
        /// </summary>
        /// <param name="state"></param>
        /// <returns></returns>
        public override Act SelectAct(Observation state)
        {
            try
            {
                var d = new SystemRandomSource();
                var v = d.NextDouble();
                var act = v < Epsilon ? Environ.SampleAct() : QTable.GetArgMax(state.Value!);
                if (act == null) throw new ArgumentException();
                return act;
            }
            catch
            {
                return Environ.SampleAct();
            }
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
            var episodes = PlayEpisode(1, PlayMode.Agent, true);
            return episodes.Length;
        }


        public Step SampleEnv()
        {
            var action = SelectAct(Environ.Observation!);
            var step = Environ.Step(action, 0);
            if (Environ.IsComplete(0)) Environ.Reset();
            return step;
        }

        public override Episode PlayEpisode(PlayMode playMode = PlayMode.Agent, bool updateAgent = false)
        {
            Environ.Reset();
            var episode = new Episode();
            var epoch = 0;
            while (Environ.IsComplete(epoch) == false)
            {
                epoch++;
                var act = playMode switch
                {
                    PlayMode.Sample => Environ.SampleAct(),
                    PlayMode.Agent => SelectAct(Environ.Observation!),
                    _ => throw new ArgumentOutOfRangeException(nameof(playMode), playMode, null)
                };
                var step = Environ.Step(act, epoch);
                if (updateAgent) Update(step);
                episode.Steps.Add(step);
                Environ.CallBack?.Invoke(step);
                Environ.Observation = step.StateNew; /// It's import for Update Observation
            }

            var orginalReward = episode.Steps.Sum(a => a.Reward.Value);
            var sumReward = orginalReward;
            episode.SumReward = new Reward(sumReward);
            return episode;
        }
    }
}
