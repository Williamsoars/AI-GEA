import os
import json
from datetime import datetime
import fcntl

class Logger:
    def __init__(self, path='recommendation_log.json'):
        self.path = path
        if not os.path.exists(self.path):
            with open(self.path, 'w') as f:
                json.dump([], f)

    def log(self, graph_name, scores, recommended):
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'graph': graph_name,
            'recommended_embedding': recommended,
            'scores': scores
        }
        
        try:
            with open(self.path, 'r+') as f:
                fcntl.flock(f, fcntl.LOCK_EX)  # Bloqueia o arquivo
                try:
                    logs = json.load(f)
                    logs.append(log_entry)
                    f.seek(0)
                    json.dump(logs, f, indent=4)
                    f.truncate()
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)  # Libera o bloqueio
        except (IOError, json.JSONDecodeError) as e:
            raise RuntimeError(f"Erro ao registrar log: {str(e)}")
