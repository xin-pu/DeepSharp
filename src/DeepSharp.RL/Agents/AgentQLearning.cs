using DeepSharp.RL.Models;
using Action = DeepSharp.RL.Models.Action;

namespace DeepSharp.RL.Agents
{
    public class AgentQLearning : Agent
    {
        public AgentQLearning(Environ env) : base(env)
        {
        }

        public override Action PredictAction(Observation reward)
        {
            throw new NotImplementedException();
        }

        public override float Learn(Episode[] steps)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        ///     奖励表
        /// </summary>
        public Dictionary<SAfS, float> Rewards { set; get; }

        /// <summary>
        ///     转移表
        /// </summary>
        public Dictionary<SA, Dictionary<torch.Tensor, int>> Transits { set; get; }

        /// <summary>
        ///     价值表
        /// </summary>
        public Dictionary<SA, float> Valus { set; get; }


        public void RunRandom(Environ environ, int count)
        {
            foreach (var i in Enumerable.Range(0, count))
            {
                var action = environ.Sample();
                var newObservation = environ.UpdateEnviron(action);
                var reward = environ.GetReward(newObservation);

                if (environ.StopEpoch(i)) environ.Reset();
            }
        }
    }

    public struct SAfS
    {
        public torch.Tensor State { set; get; }
        public long Action { set; get; }
        public torch.Tensor NewState { set; get; }
        public float Reward { set; get; }
    }

    public struct SA
    {
        public torch.Tensor State { set; get; }
        public long Action { set; get; }
    }
}