using DeepSharp.RL.Environs;

namespace DeepSharp.RL.Agents
{
    public class AgentSARSA : ValueAgent
    {
        public AgentSARSA(Environ<Space, Space> env) : base(env)
        {
        }


        public override Act GetPolicyAct(torch.Tensor state)
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