/**
 * Resolve / reload the worker-owned model for a device mode.
 * Does not run inference. On success, worker reports resolvedDevice + adapterName.
 */
export async function setDevice(
  worker,
  weightsURL,
  tokenizerURL,
  configURL,
  modelID,
  deviceMode,
  updateStatus = null
) {
  return new Promise((resolve, reject) => {
    worker.postMessage({
      command: "setDevice",
      weightsURL,
      tokenizerURL,
      configURL,
      modelID,
      deviceMode,
    });
    function messageHandler(event) {
      const data = event.data;
      if (data.status === "deviceError") {
        worker.removeEventListener("message", messageHandler);
        const err = new Error(data.error || "deviceError");
        err.deviceError = data;
        reject(err);
        return;
      }
      if ("error" in data && data.status === "error") {
        worker.removeEventListener("message", messageHandler);
        reject(new Error(data.error));
        return;
      }
      if (data.status === "ready") {
        worker.removeEventListener("message", messageHandler);
        resolve(data);
        return;
      }
      if (updateStatus) updateStatus(data);
    }
    worker.addEventListener("message", messageHandler);
  });
}

/**
 * Run embeddings on the already-loaded instance.
 * Must NOT pass deviceMode — device is fixed by setDevice.
 */
export async function getEmbeddings(
  worker,
  modelID,
  sentences,
  updateStatus = null,
  normalize = true
) {
  return new Promise((resolve, reject) => {
    worker.postMessage({
      command: "run",
      modelID,
      sentences,
      normalize,
    });
    function messageHandler(event) {
      const data = event.data;
      if (data.status === "deviceError") {
        worker.removeEventListener("message", messageHandler);
        const err = new Error(data.error || "deviceError");
        err.deviceError = data;
        reject(err);
        return;
      }
      if ("error" in data || data.status === "error") {
        worker.removeEventListener("message", messageHandler);
        reject(new Error(data.error));
        return;
      }
      if (data.status === "complete") {
        worker.removeEventListener("message", messageHandler);
        resolve(data);
        return;
      }
      if (updateStatus) updateStatus(data);
    }
    worker.addEventListener("message", messageHandler);
  });
}

const MODELS = {
  intfloat_e5_small_v2: {
    base_url: "https://huggingface.co/intfloat/e5-small-v2/resolve/main/",
    search_prefix: "query: ",
    document_prefix: "passage: ",
  },
  intfloat_e5_base_v2: {
    base_url: "https://huggingface.co/intfloat/e5-base-v2/resolve/main/",
    search_prefix: "query: ",
    document_prefix: "passage:",
  },
  intfloat_multilingual_e5_small: {
    base_url:
      "https://huggingface.co/intfloat/multilingual-e5-small/resolve/main/",
    search_prefix: "query: ",
    document_prefix: "passage: ",
  },
  sentence_transformers_all_MiniLM_L6_v2: {
    base_url:
      "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/refs%2Fpr%2F21/",
    search_prefix: "",
    document_prefix: "",
  },
  sentence_transformers_all_MiniLM_L12_v2: {
    base_url:
      "https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2/resolve/refs%2Fpr%2F4/",
    search_prefix: "",
    document_prefix: "",
  },
};
export function getModelInfo(id) {
  return {
    modelURL: MODELS[id].base_url + "model.safetensors",
    configURL: MODELS[id].base_url + "config.json",
    tokenizerURL: MODELS[id].base_url + "tokenizer.json",
    search_prefix: MODELS[id].search_prefix,
    document_prefix: MODELS[id].document_prefix,
  };
}

export function cosineSimilarity(vec1, vec2) {
  const dot = vec1.reduce((acc, val, i) => acc + val * vec2[i], 0);
  const a = Math.sqrt(vec1.reduce((acc, val) => acc + val * val, 0));
  const b = Math.sqrt(vec2.reduce((acc, val) => acc + val * val, 0));
  return dot / (a * b);
}
export async function getWikiText(article) {
  // thanks to wikipedia for the API
  const URL = `https://en.wikipedia.org/w/api.php?action=query&prop=extracts&exlimit=1&titles=${article}&explaintext=1&exsectionformat=plain&format=json&origin=*`;
  return fetch(URL, {
    method: "GET",
    headers: {
      Accept: "application/json",
    },
  })
    .then((r) => r.json())
    .then((data) => {
      const pages = data.query.pages;
      const pageId = Object.keys(pages)[0];
      const extract = pages[pageId].extract;
      if (extract === undefined || extract === "") {
        throw new Error("No article found");
      }
      return extract;
    })
    .catch((error) => console.error("Error:", error));
}
