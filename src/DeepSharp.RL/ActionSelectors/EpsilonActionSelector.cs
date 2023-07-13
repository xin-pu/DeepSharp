namespace DeepSharp.RL.ActionSelectors
{
    public class EpsilonActionSelector : ActionSelector
    {
        public EpsilonActionSelector(ActionSelector selector)
        {
            Selector = selector;
        }

        public ActionSelector Selector { protected set; get; }

        public override torch.Tensor Select(torch.Tensor probs)
        {
            throw new NotImplementedException();
        }
    }
}