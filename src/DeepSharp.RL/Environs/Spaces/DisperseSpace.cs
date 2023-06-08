namespace DeepSharp.RL.Environs.Spaces
{
    public class DisperseSpace : DigitalSpace
    {
        public DisperseSpace(long start, long length,
            DeviceType deviceType = DeviceType.CUDA, long seed = 1)
            : base(torch.tensor(new[] {start}),
                torch.tensor(new[] {start + length}),
                new long[] {1},
                torch.ScalarType.Int64, deviceType, seed)
        {
        }
    }
}