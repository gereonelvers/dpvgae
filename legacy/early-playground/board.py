from tensorboard import program

"""
Simple class to keep TensorBoard running
"""
if __name__ == '__main__':
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', './lightning_logs'])
    url = tb.launch()
    print("TensorBoard listening on "+url)
    while True:
        True
