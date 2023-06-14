using DeepSharp.RL.Environs;

namespace DeepSharp.RL.Agents
{
    public class DQN : Agent
    {
        public DQN(Environ<Space, Space> env) : base(env)
        {
        }

        public override Act SelectAct(Observation state)
        {
            throw new NotImplementedException();
        }

        public override void Update(Episode episode)
        {
            throw new NotImplementedException();
        }

        public override float Learn(int count)
        {
            throw new NotImplementedException();
        }
    }
}