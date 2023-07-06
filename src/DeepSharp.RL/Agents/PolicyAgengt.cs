using DeepSharp.RL.Environs;

namespace DeepSharp.RL.Agents
{
    public abstract class PolicyGradientAgengt : Agent
    {
        protected PolicyGradientAgengt(Environ<Space, Space> env, string name)
            : base(env, name)
        {
            PolicyNet = new PGN(ObservationSize, 128, ActionSize, DeviceType.CPU);
        }

        public Module<torch.Tensor, torch.Tensor> PolicyNet { protected set; get; }


        /// <summary>
        ///     argmax(a') Q(state,a')
        ///     价值表中获取该状态State下最高价值的action'
        /// </summary>
        /// <param name="state"></param>
        /// <returns></returns>
        public override Act GetPolicyAct(torch.Tensor state)
        {
            var probs = PolicyNet.forward(state.unsqueeze(0)).squeeze(0);
            var actIndex = torch.multinomial(probs, 1, true).ToInt32();
            return new Act(torch.from_array(new[] {actIndex}));
        }
    }
}