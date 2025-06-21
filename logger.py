import os
import json
from datetime import datetime

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
        with open(self.path, 'r+') as f:
            logs = json.load(f)
            logs.append(log_entry)
            f.seek(0)
            json.dump(logs, f, indent=4)
