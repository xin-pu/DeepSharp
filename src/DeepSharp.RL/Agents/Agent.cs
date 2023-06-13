using DeepSharp.RL.Environs;

namespace DeepSharp.RL.Agents
{
    /// <summary>
    ///     智能体
    /// </summary>
    public abstract class Agent : ObservableObject
    {
        protected Agent(Environ<Space, Space> env)
        {
            Environ = env;
            Device = env.Device;
        }

        public torch.Device Device { protected set; get; }
        public long ObservationSize => Environ.ObservationSpace.N;
        public long ActionSize => Environ.ActionSpace.N;
        public Environ<Space, Space> Environ { protected set; get; }


        public abstract Act PredictAction(Observation reward);


        public abstract float Learn(Episode[] steps);
    }
}