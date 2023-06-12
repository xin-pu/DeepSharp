namespace DeepSharp.RL.Environs.Spaces
{
    public class MultiBinary : DigitalSpace
    {
        public MultiBinary(long length, torch.ScalarType type = torch.ScalarType.Int32,
            DeviceType deviceType = DeviceType.CUDA, long seed = 471)
            : base(new[] {length}, type, deviceType, seed)
        {
        }


        public override torch.Tensor Sample()
        {
            throw new ArgumentException();
        }
    }
}