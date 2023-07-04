using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using DeepSharp.RL.Environs;

namespace DeepSharp.RL.Agents
{
    public class PolicyIteration :  Agent
    {
        public PolicyIteration(Environ<Space, Space> env, string name) 
            : base(env, "PolicyIteration")
        {
        }

        public override Act GetPolicyAct(torch.Tensor state)
        {
            throw new NotImplementedException();
        }
    }
}
