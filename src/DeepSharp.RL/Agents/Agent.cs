using DeepSharp.RL.Environs;
using DeepSharp.RL.Models;

namespace DeepSharp.RL.Agents
{
    /// <summary>
    ///     智能体
    /// </summary>
    public abstract class Agent<T1, T2> : ObservableObject
        where T1 : Space
        where T2 : Space
    {
        protected Agent(Environ<T1, T2> env)
        {
            Environ = env;
            Device = env.Device;
        }

        public torch.Device Device { protected set; get; }
        public long ObservationSize => Environ.ObservationSpace.N;
        public long ActionSize => Environ.ActionSpace.N;
        public Environ<T1, T2> Environ { protected set; get; }


        public abstract Act PredictAction(Observation reward);


        public abstract float Learn(Episode[] steps);
    }
}