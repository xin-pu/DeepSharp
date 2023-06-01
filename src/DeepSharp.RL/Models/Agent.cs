using DeepSharp.RL.Policies;

namespace DeepSharp.RL.Models
{
    /// <summary>
    ///     智能体
    /// </summary>
    public abstract class Agent : ObservableObject, IPolicy
    {
        protected Agent(Environ env)
        {
            ObservationSize = env.ObservationSpace;
            ActionSize = env.ActionSpace;
            SampleActionSpace = env.SampleActionSpace;
        }


        public int ObservationSize { protected set; get; }
        public int ActionSize { protected set; get; }
        public int SampleActionSpace { protected set; get; }


        public abstract Action PredictAction(Observation reward);


        public abstract float Learn(Step[] steps);
    }
}