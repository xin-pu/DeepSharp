using DeepSharp.RL.Models;
using DeepSharp.RL.Policies;
using Action = DeepSharp.RL.Models.Action;

namespace DeepSharp.RL.Agents
{
    /// <summary>
    ///     智能体
    /// </summary>
    public abstract class Agent : ObservableObject, IPolicy
    {
        protected Agent(Environ env)
        {
            Environ = env;
            Device = env.Device;
        }

        public torch.Device Device { protected set; get; }
        public int ObservationSize => Environ.ObservationSpace;
        public int ActionSize => Environ.ActionSpace;
        public int SampleActionSpace => Environ.SampleActionSpace;
        public Environ Environ { protected set; get; }


        public abstract Action PredictAction(Observation reward);


        public abstract float Learn(Episode[] steps);
    }
}