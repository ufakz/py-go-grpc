import logging

from concurrent.futures import ThreadPoolExecutor

import grpc
import numpy as np

from outliers_pb2 import OutliersResponse
from outliers_pb2_grpc import OutliersServicer, add_OutliersServicer_to_server


def serve():
    logging.info('server starting')
    server = grpc.server(ThreadPoolExecutor())
    add_OutliersServicer_to_server(OutliersServer(), server)
    port = 50051
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    logging.info('server ready on port %r', port)
    server.wait_for_termination()


def find_outliers(data: np.ndarray):
    out = np.where(np.abs(data - data.mean()) > 2 * data.std())
    return out[0]


class OutliersServer(OutliersServicer):
    def Detect(self, request, context):
        logging.info('detect request size: %d', len(request.metrics))
        data = np.fromiter((m.value for m in request.metrics), dtype='float64')
        indices = find_outliers(data)
        logging.info('found %d outliers', len(indices))
        resp = OutliersResponse(indices=indices)
        return resp


if __name__ == "__main__":
    serve()
