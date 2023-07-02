using DeepSharp.RL.Environs;

namespace DeepSharp.RL.Agents
{
    public class MonteCarloOffPolicy : ValueAgent
    {
        public MonteCarloOffPolicy(Environ<Space, Space> env)
            : base(env, "MonteCarloOffPolicy")
        {
        }
    }
}