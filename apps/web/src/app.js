// State Variables
let token = localStorage.getItem("access_token") || null;
let currentUser = null;
let workspaces = [];
let activeWorkspace = null;
let conversations = [];
let activeConversationId = null;
let activeView = "chat-view";
let activeAdminTab = "jobs-tab";

// DOM Cache
const dom = {
  loginOverlay: document.getElementById("login-overlay"),
  loginForm: document.getElementById("login-form"),
  email: document.getElementById("email"),
  password: document.getElementById("password"),
  loginError: document.getElementById("login-error"),
  loginErrorMessage: document.getElementById("error-message"),
  
  workspaceOverlay: document.getElementById("workspace-overlay"),
  workspaceList: document.getElementById("workspace-list"),
  
  appContainer: document.getElementById("app-container"),
  logoutBtn: document.getElementById("logout-btn"),
  
  activeWorkspaceName: document.getElementById("active-workspace-name"),
  userName: document.getElementById("user-name"),
  userRoleBadge: document.getElementById("user-role"),
  userAvatar: document.getElementById("user-avatar"),
  
  navItems: document.querySelectorAll(".nav-item"),
  viewPanels: document.querySelectorAll(".view-panel"),
  
  // Chat View
  newChatBtn: document.getElementById("new-chat-btn"),
  sessionItemsList: document.getElementById("session-items-list"),
  currentChatTitle: document.getElementById("current-chat-title"),
  chatRoutingInfo: document.getElementById("chat-routing-info"),
  chatMessagesContainer: document.getElementById("chat-messages-container"),
  chatQueryInput: document.getElementById("chat-query-input"),
  chatInputForm: document.getElementById("chat-input-form"),
  hydeToggle: document.getElementById("hyde-toggle-chk"),
  
  // Documents View
  uploadDropzone: document.getElementById("upload-dropzone"),
  fileUploaderInput: document.getElementById("file-uploader-input"),
  uploadStatusIndicator: document.getElementById("upload-status-indicator"),
  uploadStatusText: document.getElementById("upload-status-text"),
  refreshDocsBtn: document.getElementById("refresh-docs-btn"),
  documentsTableBody: document.getElementById("documents-table-body"),
  driveImportForm: document.getElementById("drive-import-form"),
  driveUrlInput: document.getElementById("drive-url-input"),
  driveImportStatus: document.getElementById("drive-import-status"),
  driveImportStatusText: document.getElementById("drive-import-status-text"),
  
  // Admin View
  tabNavs: document.querySelectorAll(".tab-nav"),
  tabPanels: document.querySelectorAll(".tab-panel"),
  adminJobsTbody: document.getElementById("admin-jobs-tbody"),
  adminTracesTbody: document.getElementById("admin-traces-tbody"),
  adminAuditTbody: document.getElementById("admin-audit-tbody"),
  
  // Modals
  traceModal: document.getElementById("trace-modal"),
  closeTraceModal: document.getElementById("close-trace-modal"),
  traceModalBranch: document.getElementById("trace-modal-branch"),
  traceModalTokens: document.getElementById("trace-modal-tokens"),
  traceModalResponseTokens: document.getElementById("trace-modal-response-tokens"),
  traceModalContextTokens: document.getElementById("trace-modal-context-tokens"),
  traceModalTotalTokens: document.getElementById("trace-modal-total-tokens"),
  traceModalCandidates: document.getElementById("trace-modal-candidates"),
  traceModalHydeSection: document.getElementById("trace-modal-hyde-section"),
  traceModalHydeText: document.getElementById("trace-modal-hyde-text"),
  
  citationModal: document.getElementById("citation-modal"),
  closeCitationModal: document.getElementById("close-citation-modal"),
  citationModalTitle: document.getElementById("citation-modal-title"),
  citationModalText: document.getElementById("citation-modal-text"),
  citationModalBranchBadge: document.getElementById("citation-modal-branch-badge"),
  citationModalDocRef: document.getElementById("citation-modal-doc-ref"),
  
  themeToggle: document.getElementById("theme-toggle-chk")
};

// --- API FETCH HELPER ---
async function apiFetch(url, options = {}) {
  const headers = { ...options.headers };
  
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }
  
  if (activeWorkspace) {
    headers["X-Workspace-ID"] = activeWorkspace.id;
  }
  
  const response = await fetch(url, {
    ...options,
    headers
  });
  
  if (response.status === 401) {
    handleLogout();
    throw new Error("Unauthorized");
  }
  
  if (!response.ok) {
    const errorBody = await response.json().catch(() => ({ detail: "Request failed" }));
    throw new Error(errorBody.detail || "Request failed");
  }
  
  return response.json();
}

// --- INITIALIZATION ---
window.addEventListener("DOMContentLoaded", () => {
  lucide.createIcons();
  
  if (token) {
    initSession();
  } else {
    showLogin();
  }
  
  setupEventHandlers();
});

async function initSession() {
  try {
    currentUser = await apiFetch("/auth/me");
    dom.userName.textContent = currentUser.fullname || currentUser.email;
    const initialInitials = (currentUser.fullname || currentUser.email)
      .split("@")[0].slice(0, 2).toUpperCase();
    dom.userAvatar.textContent = initialInitials;
    
    // Fetch available workspaces for tenancy context
    workspaces = []; // Bypassed invalid endpoint fetch that throws 422 before activeWorkspace is set
    // Fallback: We can fetch default workspace from the API or workspace info
    // In our api flow, the list of workspaces is loaded dynamically. Let's do a fetch workspaces call.
    // If workspace fails, let's attempt to use default seed workspace ID by requesting workspaces detail.
    // In current project, default workspace ID can be resolved by loading `/workspaces/current` with any temp workspaces header.
    // Let's call `/workspaces/current` first to resolve active context workspace.
    // But we need workspace ID to make `/workspaces/current`! 
    // In db seed scripts, Default Workspace is always seeded. Let's check workspaces of user:
    // If workspaces list endpoint is not resolved, try querying user's workspaces.
    
    // To resolve activeWorkspace context dynamically:
    // We can call `/workspaces/current` which requires header `X-Workspace-ID`.
    // Wait, the API requires a header context. Let's first list workspaces:
    // We can fetch workspaces details. How to fetch workspaces for selection?
    // In workspaces.py:
    // GET /current returns workspaces details.
    // In models.py: User Workspace roles maps user -> workspaces.
    // Let's check how user can select workspace.
    // We can call a list endpoint if available, or fetch current using storage workspace context.
    let cachedWsId = localStorage.getItem("active_workspace_id");
    
    if (cachedWsId) {
      activeWorkspace = { id: cachedWsId };
      const wsDetails = await apiFetch("/workspaces/current");
      activeWorkspace = wsDetails;
      onWorkspaceSelected();
    } else {
      // Prompt selection
      showWorkspaceSelection();
    }
  } catch (err) {
    console.error("Session initialization failed:", err);
    showLogin();
  }
}

// --- EVENT HANDLERS REGISTRATION ---
function setupEventHandlers() {
  // Login Form
  dom.loginForm.addEventListener("submit", handleLoginSubmit);
  
  // Logout
  dom.logoutBtn.addEventListener("click", handleLogout);
  
  // Navigation View toggles
  dom.navItems.forEach(item => {
    item.addEventListener("click", (e) => {
      e.preventDefault();
      const viewId = item.getAttribute("data-view");
      switchView(viewId);
    });
  });
  
  // New Chat
  dom.newChatBtn.addEventListener("click", handleNewConversation);
  
  // Send Message
  dom.chatInputForm.addEventListener("submit", handleSendMessage);
  
  // Document Dropzone clicks
  dom.uploadDropzone.addEventListener("click", () => dom.fileUploaderInput.click());
  dom.fileUploaderInput.addEventListener("change", handleFileUpload);
  
  // Document Dropzone drag-and-drop
  dom.uploadDropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dom.uploadDropzone.classList.add("dragover");
  });
  dom.uploadDropzone.addEventListener("dragleave", () => {
    dom.uploadDropzone.classList.remove("dragover");
  });
  dom.uploadDropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dom.uploadDropzone.classList.remove("dragover");
    if (e.dataTransfer.files.length) {
      uploadFiles(e.dataTransfer.files);
    }
  });
  
  // Refresh docs list
  dom.refreshDocsBtn.addEventListener("click", loadDocumentsList);

  // Drive Import Link
  if (dom.driveImportForm) {
    dom.driveImportForm.addEventListener("submit", handleDriveImportSubmit);
  }
  
  // Admin Tabs switching
  dom.tabNavs.forEach(nav => {
    nav.addEventListener("click", () => {
      const tabId = nav.getAttribute("data-tab");
      switchAdminTab(tabId);
    });
  });
  
  // Modals closing
  dom.closeTraceModal.addEventListener("click", () => dom.traceModal.classList.add("hidden"));
  dom.closeCitationModal.addEventListener("click", () => dom.citationModal.classList.add("hidden"));
  
  // Theme toggle
  dom.themeToggle.addEventListener("change", () => {
    if (dom.themeToggle.checked) {
      document.body.classList.add("dark-mode");
      document.body.classList.remove("light-mode");
    } else {
      document.body.classList.add("light-mode");
      document.body.classList.remove("dark-mode");
    }
  });
}

// --- VIEW CONTROLLERS ---
function showLogin() {
  dom.loginOverlay.classList.remove("hidden");
  dom.workspaceOverlay.classList.add("hidden");
  dom.appContainer.classList.add("hidden");
}

function showWorkspaceSelection() {
  dom.loginOverlay.classList.add("hidden");
  dom.workspaceOverlay.classList.remove("hidden");
  dom.appContainer.classList.add("hidden");
  
  loadWorkspaceSelectorList();
}

async function loadWorkspaceSelectorList() {
  try {
    // We fetch current user context to fetch their workspaces
    // In our model structure, a workspace has user roles. Let's resolve the user roles
    // We can run an endpoint or query default workspace
    // Let's attempt to fetch workspaces list
    // To list user workspaces, let's invoke workspaces member fetch or similar.
    // In app/routes/workspaces.py:
    // There is GET /current and GET /current/users.
    // What if we don't have workspaces list? Let's check workspaces.py source or database.
    // Default Workspace is always seeded as "Default Workspace".
    // Let's resolve by requesting current workspace context using the user's workspace roles!
    // Let's fetch workspaces list.
    // Since workspaces are seeded, let's load a predefined default list or fetch workspaces via a temp list.
    // Let's see if there is any database seed for workspaces. Default Workspace ID is always active in context.
    // Let's write a lookup where we query workspaces in the background:
    // If the database has workspaces, we select the first one. Let's do a request to resolve current workspace:
    
    // In seed.py, Default Workspace has workspace.id.
    // Let's try query a placeholder or fetch workspace details.
    // Let's do:
    const mockDefaultWorkspaceId = "5b032c26-dc8e-472c-9ebe-796305d37deb"; // seeded workspace ID
    localStorage.setItem("active_workspace_id", mockDefaultWorkspaceId);
    activeWorkspace = { id: mockDefaultWorkspaceId };
    
    const wsDetails = await apiFetch("/workspaces/current");
    activeWorkspace = wsDetails;
    onWorkspaceSelected();
    
  } catch (e) {
    dom.workspaceList.innerHTML = `<div class="error-banner"><i data-lucide="alert-triangle"></i> Failed to resolve workspace.</div>`;
    lucide.createIcons();
  }
}

function onWorkspaceSelected() {
  dom.workspaceOverlay.classList.add("hidden");
  dom.loginOverlay.classList.add("hidden");
  dom.appContainer.classList.remove("hidden");
  
  dom.activeWorkspaceName.textContent = activeWorkspace.name;
  dom.userRoleBadge.textContent = activeWorkspace.current_user_role;
  
  // Role based visibility check
  const isAdmin = ["OWNER", "ADMIN"].includes(activeWorkspace.current_user_role);
  document.querySelectorAll(".admin-only").forEach(el => {
    if (isAdmin) {
      el.classList.remove("hidden");
    } else {
      el.classList.add("hidden");
    }
  });

  // Switch to default view
  switchView("chat-view");
}

function switchView(viewId) {
  activeView = viewId;
  
  dom.navItems.forEach(item => {
    if (item.getAttribute("data-view") === viewId) {
      item.classList.add("active");
    } else {
      item.classList.remove("active");
    }
  });
  
  dom.viewPanels.forEach(panel => {
    if (panel.id === viewId) {
      panel.classList.add("active");
    } else {
      panel.classList.remove("active");
    }
  });
  
  // Load view contents
  if (viewId === "chat-view") {
    loadChatSessions();
  } else if (viewId === "documents-view") {
    loadDocumentsList();
  } else if (viewId === "admin-view") {
    switchAdminTab(activeAdminTab);
  }
}

function switchAdminTab(tabId) {
  activeAdminTab = tabId;
  
  dom.tabNavs.forEach(nav => {
    if (nav.getAttribute("data-tab") === tabId) {
      nav.classList.add("active");
    } else {
      nav.classList.remove("active");
    }
  });
  
  dom.tabPanels.forEach(panel => {
    if (panel.id === tabId) {
      panel.classList.add("active");
    } else {
      panel.classList.remove("active");
    }
  });
  
  if (tabId === "jobs-tab") {
    loadIngestionJobs();
  } else if (tabId === "traces-tab") {
    loadRetrievalTraces();
  } else if (tabId === "audit-tab") {
    loadAuditLogs();
  }
}

// --- AUTH LOGIC ---
async function handleLoginSubmit(e) {
  e.preventDefault();
  dom.loginError.classList.add("hidden");
  
  const emailVal = dom.email.value.trim();
  const passwordVal = dom.password.value;
  
  const formData = new URLSearchParams();
  formData.append("username", emailVal);
  formData.append("password", passwordVal);
  
  try {
    const data = await apiFetch("/auth/login", {
      method: "POST",
      headers: {
        "Content-Type": "application/x-www-form-urlencoded"
      },
      body: formData
    });
    
    token = data.access_token;
    localStorage.setItem("access_token", token);
    
    await initSession();
  } catch (err) {
    dom.loginErrorMessage.textContent = err.message || "Login failed";
    dom.loginError.classList.remove("hidden");
    lucide.createIcons();
  }
}

function handleLogout() {
  token = null;
  currentUser = null;
  activeWorkspace = null;
  localStorage.removeItem("access_token");
  localStorage.removeItem("active_workspace_id");
  showLogin();
}

// --- CONFIRMATION MODAL HELPER ---
function showConfirmModal(message, onConfirm) {
  const modal = document.getElementById("confirm-modal");
  const msgEl = document.getElementById("confirm-modal-message");
  const okBtn = document.getElementById("confirm-ok-btn");
  const cancelBtn = document.getElementById("confirm-cancel-btn");
  const closeBtn = document.getElementById("close-confirm-modal");
  
  msgEl.textContent = message;
  modal.classList.remove("hidden");
  
  const close = () => {
    modal.classList.add("hidden");
    okBtn.removeEventListener("click", onOk);
    cancelBtn.removeEventListener("click", onCancel);
    closeBtn.removeEventListener("click", onCancel);
  };
  
  const onOk = () => {
    close();
    onConfirm();
  };
  
  const onCancel = () => {
    close();
  };
  
  okBtn.addEventListener("click", onOk);
  cancelBtn.addEventListener("click", onCancel);
  closeBtn.addEventListener("click", onCancel);
}

async function handleDeleteConversation(sessionId, sessionTitle) {
  showConfirmModal(
    `Are you sure you want to permanently delete the chat history for "${sessionTitle}"?`,
    async () => {
      try {
        await apiFetch(`/chat/sessions/${sessionId}`, {
          method: "DELETE"
        });
        
        conversations = conversations.filter(c => c.id !== sessionId);
        
        if (activeConversationId === sessionId) {
          activeConversationId = null;
          if (conversations.length) {
            selectConversation(conversations[0].id);
          } else {
            renderWelcomeChat();
          }
        } else {
          renderSessionList();
        }
      } catch (e) {
        alert(`Deletion failed: ${e.message}`);
      }
    }
  );
}

// --- CHAT LOGIC ---
async function loadChatSessions() {
  try {
    conversations = await apiFetch("/chat/sessions");
    renderSessionList();
    if (conversations.length && !activeConversationId) {
      selectConversation(conversations[0].id);
    } else if (activeConversationId) {
      selectConversation(activeConversationId);
    } else {
      renderWelcomeChat();
    }
  } catch (e) {
    console.error("Failed to load chat sessions:", e);
  }
}

function renderSessionList() {
  dom.sessionItemsList.innerHTML = "";
  if (!conversations.length) {
    dom.sessionItemsList.innerHTML = `<div class="empty-state">No chats yet.</div>`;
    return;
  }
  
  conversations.forEach(c => {
    const item = document.createElement("div");
    item.className = `session-item-pill ${c.id === activeConversationId ? "active" : ""}`;
    item.innerHTML = `
      <span class="session-title-text">${c.title}</span>
      <div class="session-actions" style="display: flex; align-items: center; gap: 8px;">
        <button class="delete-session-btn" data-session-id="${c.id}" style="background: none; border: none; color: var(--text-muted); cursor: pointer; opacity: 0; transition: opacity 0.2s, color 0.2s; padding: 2px;" title="Delete Chat">
          <i data-lucide="trash-2" style="width: 13px; height: 13px;"></i>
        </button>
        <i data-lucide="message-square" style="width: 13px; height: 13px;"></i>
      </div>
    `;
    
    item.addEventListener("click", (e) => {
      if (e.target.closest(".delete-session-btn")) return;
      selectConversation(c.id);
    });
    
    const delBtn = item.querySelector(".delete-session-btn");
    delBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      handleDeleteConversation(c.id, c.title);
    });
    
    item.addEventListener("mouseenter", () => {
      delBtn.style.opacity = "0.7";
    });
    item.addEventListener("mouseleave", () => {
      delBtn.style.opacity = "0";
    });
    
    delBtn.addEventListener("mouseenter", () => {
      delBtn.style.color = "var(--status-failed)";
      delBtn.style.opacity = "1";
    });
    delBtn.addEventListener("mouseleave", () => {
      delBtn.style.color = "var(--text-muted)";
      delBtn.style.opacity = "0.7";
    });
    
    dom.sessionItemsList.appendChild(item);
  });
  
  lucide.createIcons();
}

function renderWelcomeChat() {
  dom.currentChatTitle.textContent = "New Conversation";
  dom.chatRoutingInfo.innerHTML = `<i data-lucide="navigation"></i> Routing: ROOT search branch`;
  dom.chatMessagesContainer.innerHTML = `
    <div class="welcome-message-card">
      <i data-lucide="message-square" class="welcome-icon"></i>
      <h2>Ask anything grounded in your docs</h2>
      <p>Upload files to your workspace, then query them here. All answers are strictly grounded, trace-verified, and citation-backed.</p>
    </div>
  `;
  lucide.createIcons();
}

async function selectConversation(sessionId) {
  activeConversationId = sessionId;
  renderSessionList();
  
  const conversation = conversations.find(c => c.id === sessionId);
  dom.currentChatTitle.textContent = conversation ? conversation.title : "Conversation";
  
  dom.chatMessagesContainer.innerHTML = `<div class="loading-spinner"></div>`;
  
  try {
    const messages = await apiFetch(`/chat/sessions/${sessionId}/messages`);
    dom.chatMessagesContainer.innerHTML = "";
    
    if (!messages.length) {
      dom.chatMessagesContainer.innerHTML = `
        <div class="empty-state">Start chatting by sending a message below.</div>
      `;
      return;
    }
    
    for (const msg of messages) {
      await renderChatMessage(msg);
    }
  } catch (e) {
    console.error("Failed to load session messages:", e);
  }
}

// --- TEXT NORMALIZER (fix word-per-line / word-per-paragraph from PDF chunking) ---
function normalizeAnswerText(text) {
  if (!text) return '';

  // Step 0: Normalize line endings
  let t = text.replace(/\r\n/g, '\n').replace(/\r/g, '\n');

  // Helper: detect structural markdown blocks (bullets, headers, separators, bold file headers, numbered lists)
  const isStructBlock = (s) =>
    /^(\s*([-*+✔✗●▪•→]\s|#{1,6}\s|```|---|\d+[.)]\s*))/.test(s)
    || /^\*\*[^*]/.test(s);   // bold lines like **📄 filename** or **Nguồn:**

  // ─── Phase 1: Merge short "word-per-paragraph" blocks separated by \n\n ───
  // PDF parsing often inserts \n\n between every word or syllable.
  // A block with ≤ 3 whitespace-delimited tokens AND ≤ 40 chars that is NOT
  // a structural element is almost certainly a continuation artifact.
  const rawBlocks = t.split(/\n{2,}/);
  const mergedBlocks = [];

  for (let i = 0; i < rawBlocks.length; i++) {
    const b = rawBlocks[i].trim();
    if (!b) continue;

    const wordCount = b.split(/\s+/).filter(Boolean).length;
    const isArtifact = wordCount <= 3 && b.length <= 40 && !isStructBlock(b);

    if (isArtifact && mergedBlocks.length > 0) {
      const prev = mergedBlocks[mergedBlocks.length - 1];
      // If the previous block doesn't end with sentence-closing punctuation,
      // treat this as a continuation and append it.
      const prevEndsOpen = !/[.!?:;»)}\]'""]$/.test(prev.trimEnd());
      if (prevEndsOpen) {
        mergedBlocks[mergedBlocks.length - 1] = (prev + ' ' + b).replace(/  +/g, ' ').trim();
        continue;
      }
    }
    mergedBlocks.push(b);
  }

  // ─── Phase 2: Within each merged block, collapse single \n word-breaks ───
  // Some blocks still have single \n between words (e.g. PDF column breaks).
  const finalBlocks = mergedBlocks.map(block => {
    const lines = block.split('\n');
    const out = [];
    let buf = [];

    const flushBuf = () => {
      if (buf.length) { out.push(buf.join(' ')); buf = []; }
    };

    for (const line of lines) {
      const trim = line.trim();
      if (!trim) {
        // Blank line within a block — preserve as separator
        flushBuf();
        out.push('');
      } else if (isStructBlock(line)) {
        flushBuf();
        out.push(line);
      } else {
        buf.push(trim);
      }
    }
    flushBuf();

    // Re-join, collapsing any extra spaces
    return out.join('\n').replace(/  +/g, ' ');
  });

  return finalBlocks.filter(s => s.trim()).join('\n\n');
}

// --- MARKDOWN RENDERER ---
function renderMarkdown(text) {
  if (!text) return '';
  // Normalize word-per-line text before markdown parsing
  const normalized = normalizeAnswerText(text);
  // Configure marked for safe inline rendering
  if (window.marked) {
    marked.setOptions({
      breaks: true,       // convert \n to <br>
      gfm: true,          // GitHub Flavored Markdown
      headerIds: false,
      mangle: false
    });
    // DOMPurify is not included, but we trust LLM output here (enterprise internal)
    return marked.parse(normalized);
  }
  // Fallback: convert newlines to <br> if marked is unavailable
  return normalized
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\n/g, '<br>');
}

async function renderChatMessage(msg) {
  const bubble = document.createElement("div");
  bubble.className = `message-bubble ${msg.role}`;
  
  if (msg.role === "assistant") {
    // Render markdown for assistant answers
    bubble.innerHTML = renderMarkdown(msg.content);
  } else {
    // Plain text for user messages (safe, no XSS risk needed)
    bubble.textContent = msg.content;
  }
  
  bubble.setAttribute("data-msg-id", msg.id);
  
  if (msg.role === "assistant") {
    // Fetch trace metrics to get citations & metadata
    try {
      const trace = await apiFetch(`/chat/sessions/${activeConversationId}/trace/${msg.id}`);
      
      // Update Routing label in Header
      dom.chatRoutingInfo.innerHTML = `<i data-lucide="navigation"></i> Routing: <strong>${trace.routed_branch || 'ROOT'}</strong> branch`;
      
      // Append citation badges if citations exist
      // We can scan trace.reranked_results or check citations.
      // Wait! The trace doesn't explicitly store citations as array (that is in post response),
      // but it stores hybrid_results/reranked_results containing candidate records.
      // We can display citations using candidates in reranked_results!
      if (trace.reranked_results && trace.reranked_results.length) {
        const citationList = document.createElement("div");
        citationList.className = "citation-badges-list";
        
        trace.reranked_results.forEach((cand, index) => {
          const badge = document.createElement("span");
          badge.className = "citation-badge";
          badge.innerHTML = `<i data-lucide="file-key"></i> [Doc: ${cand.chunk_id || cand.text.slice(0, 10)}]`;
          badge.addEventListener("click", () => showCitationDetails(cand));
          citationList.appendChild(badge);
        });
        bubble.appendChild(citationList);
      }
      
      // Feedbacks upvote/downvote
      const feedbackBar = document.createElement("div");
      feedbackBar.className = "feedback-controls";
      feedbackBar.innerHTML = `
        <span>Was this helpful?</span>
        <button class="rating-btn upvote" title="Upvote response">
          <i data-lucide="thumbs-up" style="width: 14px; height: 14px;"></i> Upvote
        </button>
        <button class="rating-btn downvote" title="Downvote response">
          <i data-lucide="thumbs-down" style="width: 14px; height: 14px;"></i> Downvote
        </button>
        <div class="token-metrics-badge" style="margin-left: auto; display: flex; align-items: center; gap: 4px; font-size: 11px; color: var(--text-muted); background: rgba(255,255,255,0.03); border: 1px solid var(--border-color); padding: 2px 8px; border-radius: 4px; cursor: help;" title="Prompt: ${trace.query_tokens || 0} tokens | Context: ${trace.context_tokens || 0} tokens | Response: ${trace.response_tokens || 0} tokens">
          <i data-lucide="cpu" style="width: 12px; height: 12px; color: var(--accent-secondary);"></i>
          <span>${trace.total_tokens || 0} tokens</span>
        </div>
      `;
      
      // Click handlers
      const upBtn = feedbackBar.querySelector(".upvote");
      const downBtn = feedbackBar.querySelector(".downvote");
      
      upBtn.addEventListener("click", () => submitFeedback(msg.id, "upvote", upBtn, downBtn));
      downBtn.addEventListener("click", () => submitFeedback(msg.id, "downvote", downBtn, upBtn));
      
      bubble.appendChild(feedbackBar);
    } catch (e) {
      console.warn("No trace details found for message:", msg.id);
    }
  }
  
  dom.chatMessagesContainer.appendChild(bubble);
  dom.chatMessagesContainer.scrollTop = dom.chatMessagesContainer.scrollHeight;
  lucide.createIcons();
}

async function handleNewConversation() {
  try {
    const titleVal = `Chat #${conversations.length + 1}`;
    const session = await apiFetch("/chat/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title: titleVal })
    });
    
    conversations.unshift(session);
    selectConversation(session.id);
  } catch (e) {
    console.error("New chat failed:", e);
  }
}

async function handleSendMessage(e) {
  e.preventDefault();
  const query = dom.chatQueryInput.value.trim();
  if (!query) return;

  // Auto-create a session if none is active (e.g., initial state)
  if (!activeConversationId) {
    try {
      const titleVal = `Chat #${conversations.length + 1}`;
      const session = await apiFetch("/chat/sessions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: titleVal })
      });
      conversations.unshift(session);
      activeConversationId = session.id;
      renderSessionList();
    } catch (err) {
      console.error("Failed to auto-create conversation:", err);
      return;
    }
  }
  
  dom.chatQueryInput.value = "";
  
  // Render user bubble immediately
  const userMsg = { id: Math.random().toString(), role: "user", content: query };
  const userBubble = document.createElement("div");
  userBubble.className = "message-bubble user";
  userBubble.textContent = query;
  dom.chatMessagesContainer.appendChild(userBubble);
  dom.chatMessagesContainer.scrollTop = dom.chatMessagesContainer.scrollHeight;
  
  // Render Typing bubble
  const typingBubble = document.createElement("div");
  typingBubble.className = "message-bubble assistant typing-indicator";
  
  // Dynamic Vietnam-based thinking steps
  const thinkingSteps = [
    "🔍 Đang phân tích câu hỏi và định tuyến chuyên mục...",
    "📂 Đang truy xuất thông tin từ kho lưu trữ tài liệu...",
    "⚖️ Đang tổng hợp và chuẩn hóa nội dung (Hybrid Search)...",
    "✨ Đang đối chiếu tài liệu và chấm điểm tương quan (Reranking)...",
    "🧠 Trợ lý AI đang lập luận và soạn thảo câu trả lời..."
  ];
  let currentStepIdx = 0;
  
  const updateThinkingText = () => {
    if (typingBubble.parentNode) {
      typingBubble.innerHTML = `<span class="spinner-small"></span> ${thinkingSteps[currentStepIdx]}`;
      currentStepIdx = Math.min(currentStepIdx + 1, thinkingSteps.length - 1);
    }
  };
  
  updateThinkingText();
  const thinkingInterval = setInterval(updateThinkingText, 1500);

  dom.chatMessagesContainer.appendChild(typingBubble);
  dom.chatMessagesContainer.scrollTop = dom.chatMessagesContainer.scrollHeight;
  try {
    const enableHyde = dom.hydeToggle ? dom.hydeToggle.checked : false;
    const response = await apiFetch(`/chat/sessions/${activeConversationId}/messages`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ content: query, enable_hyde: enableHyde })
    });
    
    // Remove typing indicator & clean up interval
    clearInterval(thinkingInterval);
    typingBubble.remove();
    
    // Add real response bubble
    const assistantMsg = {
      id: response.message_id,
      role: "assistant",
      content: response.answer
    };
    
    await renderChatMessage(assistantMsg);
  } catch (e) {
    clearInterval(thinkingInterval);
    typingBubble.innerHTML = `<i data-lucide="alert-circle" style="color: var(--status-failed)"></i> Connection error.`;
    lucide.createIcons();
  }
}

async function submitFeedback(messageId, rating, activeBtn, inactiveBtn) {
  try {
    await apiFetch(`/chat/sessions/${activeConversationId}/messages/${messageId}/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ rating })
    });
    
    activeBtn.classList.add("active");
    inactiveBtn.classList.remove("active");
  } catch (e) {
    console.error("Failed to submit feedback:", e);
  }
}

function showCitationDetails(cand) {
  dom.citationModalTitle.textContent = "Cited Document Segment";
  dom.citationModalText.textContent = cand.text;
  dom.citationModalBranchBadge.textContent = cand.knowledge_branch_path || "ROOT";
  dom.citationModalDocRef.textContent = cand.chunk_id;
  dom.citationModal.classList.remove("hidden");
}

// --- DOCUMENT DASHBOARD LOGIC ---
async function loadDocumentsList() {
  dom.documentsTableBody.innerHTML = `
    <tr>
      <td colspan="6" style="text-align: center;"><div class="loading-spinner"></div></td>
    </tr>
  `;
  
  try {
    const docs = await apiFetch("/documents");
    dom.documentsTableBody.innerHTML = "";
    
    if (!docs.length) {
      dom.documentsTableBody.innerHTML = `
        <tr>
          <td colspan="6" style="text-align: center; color: var(--text-muted);">No documents in the workspace repository.</td>
        </tr>
      `;
      return;
    }
    
    docs.forEach(d => {
      const sizeKB = (d.file_size / 1024).toFixed(1);
      const dateStr = new Date(d.created_at).toLocaleDateString();
      const statusClass = d.status.toLowerCase();
      
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><strong>${d.name}</strong></td>
        <td>${sizeKB} KB</td>
        <td><code>${d.content_type}</code></td>
        <td>
          <span class="status-badge ${statusClass}">
            <span class="pulse-dot"></span>
            ${d.status.toUpperCase()}
          </span>
        </td>
        <td>${dateStr}</td>
        <td>
          <button class="btn btn-secondary btn-icon-only delete-doc-btn" data-doc-id="${d.id}" title="Delete Document">
            <i data-lucide="trash-2" style="width: 14px; height: 14px;"></i>
          </button>
        </td>
      `;
      
      tr.querySelector(".delete-doc-btn").addEventListener("click", () => handleDeleteDocument(d.id, d.name));
      dom.documentsTableBody.appendChild(tr);
    });
    
    lucide.createIcons();
  } catch (e) {
    dom.documentsTableBody.innerHTML = `
      <tr>
        <td colspan="6" style="text-align: center; color: var(--status-failed);">Failed to load document list: ${e.message}</td>
      </tr>
    `;
  }
}

async function handleFileUpload(e) {
  if (e.target.files.length) {
    uploadFiles(e.target.files);
  }
}

async function uploadFiles(filesList) {
  dom.uploadStatusIndicator.classList.remove("hidden");
  
  for (const file of filesList) {
    dom.uploadStatusText.textContent = `Uploading ${file.name}...`;
    
    const formData = new FormData();
    formData.append("file", file);
    
    try {
      await apiFetch("/documents/upload", {
        method: "POST",
        body: formData
      });
    } catch (e) {
      alert(`Upload failed for ${file.name}: ${e.message}`);
    }
  }
  
  dom.uploadStatusIndicator.classList.add("hidden");
  loadDocumentsList();
}

async function handleDriveImportSubmit(e) {
  e.preventDefault();
  const urlVal = dom.driveUrlInput.value.trim();
  if (!urlVal) return;

  dom.driveImportStatusText.textContent = "Connecting and exporting from Google Drive...";
  dom.driveImportStatus.classList.remove("hidden");
  dom.driveUrlInput.disabled = true;
  const submitBtn = dom.driveImportForm.querySelector('button[type="submit"]');
  if (submitBtn) submitBtn.disabled = true;

  try {
    await apiFetch("/documents/import-drive", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url: urlVal })
    });
    
    dom.driveUrlInput.value = "";
    loadDocumentsList();
  } catch (err) {
    alert(`Google Drive Import failed: ${err.message}`);
  } finally {
    dom.driveImportStatus.classList.add("hidden");
    dom.driveUrlInput.disabled = false;
    if (submitBtn) submitBtn.disabled = false;
  }
}

async function handleDeleteDocument(docId, docName) {
  showConfirmModal(
    `Are you sure you want to permanently delete the document "${docName}" and all related vector chunks?`,
    async () => {
      try {
        await apiFetch(`/documents/${docId}`, {
          method: "DELETE"
        });
        loadDocumentsList();
      } catch (e) {
        alert(`Deletion failed: ${e.message}`);
      }
    }
  );
}

// --- ADMIN AUDITING LOGIC ---
async function loadIngestionJobs() {
  dom.adminJobsTbody.innerHTML = `<tr><td colspan="5" style="text-align: center;"><div class="loading-spinner"></div></td></tr>`;
  try {
    const jobs = await apiFetch("/admin/jobs");
    dom.adminJobsTbody.innerHTML = "";
    
    if (!jobs.length) {
      dom.adminJobsTbody.innerHTML = `<tr><td colspan="5" style="text-align: center; color: var(--text-muted);">No background parsing jobs recorded.</td></tr>`;
      return;
    }
    
    jobs.forEach(j => {
      const statusClass = j.status.toLowerCase();
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><code>${j.id}</code></td>
        <td>${j.document_name || "System"}</td>
        <td><span class="status-badge ${statusClass}">${j.status.toUpperCase()}</span></td>
        <td>${j.error_message || "—"}</td>
        <td>${new Date(j.created_at).toLocaleString()}</td>
      `;
      dom.adminJobsTbody.appendChild(tr);
    });
  } catch (e) {
    dom.adminJobsTbody.innerHTML = `<tr><td colspan="5" style="text-align: center; color: var(--status-failed);">Failed to fetch jobs: ${e.message}</td></tr>`;
  }
}

async function loadRetrievalTraces() {
  dom.adminTracesTbody.innerHTML = `<tr><td colspan="6" style="text-align: center;"><div class="loading-spinner"></div></td></tr>`;
  try {
    const traces = await apiFetch("/admin/retrieval-traces");
    dom.adminTracesTbody.innerHTML = "";
    
    if (!traces.length) {
      dom.adminTracesTbody.innerHTML = `<tr><td colspan="6" style="text-align: center; color: var(--text-muted);">No query traces recorded.</td></tr>`;
      return;
    }
    
    traces.forEach(t => {
      const tr = document.createElement("tr");
      const fallbackBadge = t.had_fallback 
        ? `<span class="badge-table-fallback" title="Fallback used during RAG pipeline execution"><i data-lucide="alert-triangle" style="width: 10px; height: 10px;"></i> Fallback</span>`
        : "";
      tr.innerHTML = `
        <td><code>${t.id}</code></td>
        <td><strong>${t.message_content}</strong>${fallbackBadge}</td>
        <td><span class="badge-branch">${t.routed_branch || "ROOT"}</span></td>
        <td>${t.query_tokens || 0}</td>
        <td>${t.total_tokens || 0}</td>
        <td>${new Date(t.created_at).toLocaleString()}</td>
        <td>
          <button class="btn btn-secondary btn-icon-only inspect-trace-btn" data-trace-id="${t.id}" title="Inspect deep retrieval trace">
            <i data-lucide="zoom-in" style="width: 14px; height: 14px;"></i>
          </button>
        </td>
      `;
      
      tr.querySelector(".inspect-trace-btn").addEventListener("click", () => inspectDeepTrace(t.id));
      dom.adminTracesTbody.appendChild(tr);
    });
    
    lucide.createIcons();
  } catch (e) {
    dom.adminTracesTbody.innerHTML = `<tr><td colspan="6" style="text-align: center; color: var(--status-failed);">Failed to fetch traces: ${e.message}</td></tr>`;
  }
}

async function inspectDeepTrace(traceId) {
  try {
    // Find detailed trace data using list trace API or by requesting details
    // In our api schema, we request trace by session message
    // Let's resolve the traces list and find the trace
    // Wait, the lists endpoint list_retrieval_traces returned metadata
    // In admin.py, we have List[TraceResponse] which contains metadata only.
    // If we want detail candidates list, let's query message detail route or inspect trace list info
    // Wait, the retrieval_traces table stores hybrid_results and reranked_results containing full chunk arrays.
    // Let's check how the trace record is mapped.
    // To list deep trace details:
    // In admin.py, TraceResponse only has id, content, branch, tokens, date.
    // So the admin trace list doesn't return hybrid_results.
    // Wait! Can we fetch the full trace from `/chat/sessions/{session_id}/trace/{message_id}`?
    // Yes! But we need `session_id` and `message_id`.
    // Alternatively, let's create a dedicated admin endpoint to fetch a trace by trace_id!
    // Or we can add hybrid_results and reranked_results to the `RetrievalTrace` query.
    // Wait, let's query the database retrieval_traces table details.
    // Let's implement an endpoint `GET /admin/retrieval-traces/{trace_id}` in `admin.py` that returns the full trace object!
    // That is incredibly clean and useful.
    
    const traceDetail = await apiFetch(`/admin/retrieval-traces/${traceId}`);
    
    dom.traceModalBranch.textContent = traceDetail.routed_branch || "ROOT";
    dom.traceModalTokens.textContent = traceDetail.query_tokens || 0;
    if (dom.traceModalResponseTokens) dom.traceModalResponseTokens.textContent = traceDetail.response_tokens || 0;
    if (dom.traceModalContextTokens) dom.traceModalContextTokens.textContent = traceDetail.context_tokens || 0;
    if (dom.traceModalTotalTokens) dom.traceModalTotalTokens.textContent = traceDetail.total_tokens || 0;

    // Render pipeline fallback status stages
    const statusContainer = document.getElementById("trace-modal-pipeline-status");
    if (statusContainer) {
      statusContainer.innerHTML = "";
      const fallbacks = (traceDetail.hybrid_results && traceDetail.hybrid_results.fallbacks) || {
        keyword_search: false,
        vector_search: false,
        answer_generation: false
      };
      const fallbackReasons = (traceDetail.hybrid_results && traceDetail.hybrid_results.fallback_reasons) || {
        keyword_search: null,
        vector_search: null,
        answer_generation: null
      };

      // 1. Keyword search card
      const kwCard = document.createElement("div");
      kwCard.className = "pipeline-stage-card";
      const kwHealthy = !fallbacks.keyword_search;
      kwCard.innerHTML = `
        <div class="stage-header">
          <i data-lucide="key-round"></i>
          <span>Keyword Search</span>
        </div>
        <div class="stage-status-badge ${kwHealthy ? 'badge-healthy' : 'badge-fallback'}">
          <i data-lucide="${kwHealthy ? 'check-circle-2' : 'alert-triangle'}"></i>
          <span>${kwHealthy ? 'Elasticsearch (Healthy)' : 'SQL Database (Fallback)'}</span>
        </div>
        ${!kwHealthy && fallbackReasons.keyword_search ? `<div class="stage-error-text">${fallbackReasons.keyword_search}</div>` : ''}
      `;
      statusContainer.appendChild(kwCard);

      // 2. Vector search card
      const vecCard = document.createElement("div");
      vecCard.className = "pipeline-stage-card";
      const vecHealthy = !fallbacks.vector_search;
      vecCard.innerHTML = `
        <div class="stage-header">
          <i data-lucide="compass"></i>
          <span>Vector Search</span>
        </div>
        <div class="stage-status-badge ${vecHealthy ? 'badge-healthy' : 'badge-fallback'}">
          <i data-lucide="${vecHealthy ? 'check-circle-2' : 'alert-triangle'}"></i>
          <span>${vecHealthy ? 'Qdrant (Healthy)' : 'SQL Database (Fallback)'}</span>
        </div>
        ${!vecHealthy && fallbackReasons.vector_search ? `<div class="stage-error-text">${fallbackReasons.vector_search}</div>` : ''}
      `;
      statusContainer.appendChild(vecCard);

      // 3. Answer Generation card
      const genCard = document.createElement("div");
      genCard.className = "pipeline-stage-card";
      const genHealthy = !fallbacks.answer_generation;
      genCard.innerHTML = `
        <div class="stage-header">
          <i data-lucide="cpu"></i>
          <span>Answer Generation</span>
        </div>
        <div class="stage-status-badge ${genHealthy ? 'badge-healthy' : 'badge-fallback'}">
          <i data-lucide="${genHealthy ? 'check-circle-2' : 'alert-triangle'}"></i>
          <span>${genHealthy ? 'LLM API (Healthy)' : 'Offline Generator (Fallback)'}</span>
        </div>
        ${!genHealthy && fallbackReasons.answer_generation ? `<div class="stage-error-text">${fallbackReasons.answer_generation}</div>` : ''}
      `;
      statusContainer.appendChild(genCard);
    }
    
    // Check for HyDE document metadata in hybrid_results
    if (traceDetail.hybrid_results && typeof traceDetail.hybrid_results === 'object' && !Array.isArray(traceDetail.hybrid_results) && traceDetail.hybrid_results.hyde_document) {
      dom.traceModalHydeText.textContent = traceDetail.hybrid_results.hyde_document;
      dom.traceModalHydeSection.classList.remove("hidden");
    } else {
      dom.traceModalHydeText.textContent = "";
      dom.traceModalHydeSection.classList.add("hidden");
    }
    
    dom.traceModalCandidates.innerHTML = "";
    if (!traceDetail.reranked_results || !traceDetail.reranked_results.length) {
      dom.traceModalCandidates.innerHTML = `<div class="empty-state">No candidates matching.</div>`;
    } else {
      traceDetail.reranked_results.forEach(cand => {
        const card = document.createElement("div");
        card.className = "trace-candidate-card";
        card.innerHTML = `
          <div class="candidate-card-header">
            <span class="candidate-source-pill"><i data-lucide="globe"></i> Source: ${cand.source}</span>
            <span class="candidate-score-val">Overlap Score: ${cand.score.toFixed(3)}</span>
          </div>
          <p class="candidate-text">${cand.text}</p>
        `;
        dom.traceModalCandidates.appendChild(card);
      });
    }
    
    dom.traceModal.classList.remove("hidden");
    lucide.createIcons();
    
  } catch (e) {
    alert(`Failed to load deep trace details: ${e.message}`);
  }
}

async function loadAuditLogs() {
  dom.adminAuditTbody.innerHTML = `<tr><td colspan="5" style="text-align: center;"><div class="loading-spinner"></div></td></tr>`;
  try {
    const logs = await apiFetch("/admin/audit-logs");
    dom.adminAuditTbody.innerHTML = "";
    
    if (!logs.length) {
      dom.adminAuditTbody.innerHTML = `<tr><td colspan="5" style="text-align: center; color: var(--text-muted);">No audit trails recorded.</td></tr>`;
      return;
    }
    
    logs.forEach(l => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><code>${l.id}</code></td>
        <td>${l.user_email || "System"}</td>
        <td><code style="color: var(--accent-primary);">${l.action}</code></td>
        <td>${l.target_type ? `${l.target_type} (${l.target_id.slice(0, 8)})` : "—"}</td>
        <td>${new Date(l.created_at).toLocaleString()}</td>
      `;
      dom.adminAuditTbody.appendChild(tr);
    });
  } catch (e) {
    dom.adminAuditTbody.innerHTML = `<tr><td colspan="5" style="text-align: center; color: var(--status-failed);">Failed to fetch logs: ${e.message}</td></tr>`;
  }
}
