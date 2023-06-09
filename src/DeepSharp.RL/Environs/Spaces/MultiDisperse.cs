namespace DeepSharp.RL.Environs.Spaces
{
    /// <summary>
    ///     Todo
    /// </summary>
    public class MultiDisperse : DigitalSpace
    {
        public MultiDisperse(torch.Tensor low, torch.Tensor high, long[] shape, torch.ScalarType type,
            DeviceType deviceType = DeviceType.CUDA, long seed = 1) : base(low, high, shape, type, deviceType, seed)
        {
        }

        public override torch.Tensor Sample()
        {
            throw new NotImplementedException();
        }

        public override void CheckType()
        {
            throw new NotImplementedException();
        }
    }
}