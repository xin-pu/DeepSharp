using DeepSharp.RL.Environs;

namespace DeepSharp.RL.Agents
{
    public class MonteCarlo : ValueAgent
    {
        public MonteCarlo(Environ<Space, Space> env, string name)
            : base(env, "MonteCarlo")
        {
        }
    }
}