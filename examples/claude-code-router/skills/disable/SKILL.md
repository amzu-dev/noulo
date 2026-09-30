---
description: Turn noulo-router's routing off in every session until /noulo-router:enable. Status, why and correct keep working.
disable-model-invocation: true
---

noulo-router's hook answers this command before it reaches you, so you shouldn't be reading
this. If you are, the hook didn't run: tell the user in one sentence that noulo-router's hook
isn't working (check that `python3` runs and that the plugin is enabled), and do nothing else.
