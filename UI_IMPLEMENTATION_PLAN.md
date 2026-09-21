# Inference Matrix - UI Implementation Plan

## Phase 1: Complete Core Features (Week 1-2)

### 1.1 Agents Management UI ✋ TODO
**Priority**: High | **Estimated**: 2 days

**Files to Create:**
- `frontend/src/components/Agents/AgentsList.tsx`
- `frontend/src/components/Agents/AddAgent.tsx`
- `frontend/src/components/Agents/EditAgent.tsx`
- `frontend/src/components/Agents/DeleteAgent.tsx`
- `frontend/src/components/Agents/AgentStatus.tsx`
- `frontend/src/components/Agents/columns.tsx`
- `frontend/src/components/Agents/AgentActionsMenu.tsx`

**Backend APIs Needed:**
- [x] `GET /api/v1/agents/` - List agents
- [ ] `POST /api/v1/agents/` - Create agent
- [ ] `PUT /api/v1/agents/{id}` - Update agent
- [ ] `DELETE /api/v1/agents/{id}` - Delete agent
- [ ] `GET /api/v1/agents/{id}/status` - Get agent status
- [ ] `GET /api/v1/agents/{id}/logs` - Get agent logs

**UI Features:**
- [ ] Data table with agent list
- [ ] Status indicators (online/offline/busy)
- [ ] Add agent dialog with form validation
- [ ] Edit agent configuration
- [ ] Delete confirmation
- [ ] Real-time status updates (WebSocket)
- [ ] Logs viewer modal

---

### 1.2 Server Instances UI ✋ TODO
**Priority**: High | **Estimated**: 2 days

**Files to Create:**
- `frontend/src/components/Servers/ServerList.tsx`
- `frontend/src/components/Servers/AddServer.tsx`
- `frontend/src/components/Servers/EditServer.tsx`
- `frontend/src/components/Servers/DeleteServer.tsx`
- `frontend/src/components/Servers/ServerHealth.tsx`
- `frontend/src/components/Servers/columns.tsx`
- `frontend/src/components/Servers/ServerActionsMenu.tsx`

**Backend APIs Needed:**
- [x] `GET /api/v1/servers/` - List servers
- [ ] `POST /api/v1/servers/` - Create server
- [ ] `PUT /api/v1/servers/{id}` - Update server
- [ ] `DELETE /api/v1/servers/{id}` - Delete server
- [ ] `GET /api/v1/servers/{id}/health` - Health check
- [ ] `POST /api/v1/servers/{id}/test-connection` - Test connection

**UI Features:**
- [ ] Server instances table
- [ ] Health status badges
- [ ] Add server form (URL, config)
- [ ] Connection test button
- [ ] Edit server configuration
- [ ] Delete with confirmation
- [ ] Auto-refresh health status

---

### 1.3 Settings Page ✋ TODO
**Priority**: Medium | **Estimated**: 1 day

**Files to Update:**
- `frontend/src/routes/_layout/settings.tsx`
- `frontend/src/components/Settings/`

**Sections to Implement:**
- [ ] **Preferences**
  - Theme (dark/light/system)
  - Language
  - Default model
- [ ] **Notifications**
  - Email notifications
  - Browser notifications

---

## Phase 2: Inference Features (Week 3-4)

### 2.1 Chat Completion UI ✋ TODO
**Priority**: High | **Estimated**: 3 days

**Files to Create:**
- `frontend/src/routes/_layout/chat.tsx`
- `frontend/src/components/Chat/ChatInterface.tsx`
- `frontend/src/components/Chat/MessageList.tsx`
- `frontend/src/components/Chat/MessageInput.tsx`
- `frontend/src/components/Chat/ModelSelector.tsx`
- `frontend/src/components/Chat/ChatSettings.tsx`
- `frontend/src/components/Chat/ConversationHistory.tsx`

**Backend APIs:**
- [x] `POST /api/v1/chat/completions` - Chat completion
- [ ] `GET /api/v1/chat/conversations` - List conversations
- [ ] `POST /api/v1/chat/conversations` - Create conversation
- [ ] `DELETE /api/v1/chat/conversations/{id}` - Delete conversation

**UI Features:**
- [ ] Chat interface with message bubbles
- [ ] Model selection dropdown
- [ ] System prompt configuration
- [ ] Temperature/top_p sliders
- [ ] Conversation history sidebar
- [ ] Export conversation (JSON, TXT, MD)
- [ ] Clear conversation
- [ ] Streaming responses
- [ ] Code syntax highlighting
- [ ] Markdown rendering

---

### 2.2 Text Completion UI ✋ TODO
**Priority**: Medium | **Estimated**: 2 days

**Files to Create:**
- `frontend/src/routes/_layout/completions.tsx`
- `frontend/src/components/Completions/CompletionInterface.tsx`
- `frontend/src/components/Completions/PromptEditor.tsx`
- `frontend/src/components/Completions/CompletionSettings.tsx`

**Backend APIs:**
- [x] `POST /api/v1/completions` - Text completion

**UI Features:**
- [ ] Text editor with prompt
- [ ] Completion settings (max tokens, temperature, etc.)
- [ ] Generate button
- [ ] Display completion result
- [ ] Copy to clipboard
- [ ] Save to prompts library

---

### 2.3 Embeddings UI ✋ TODO
**Priority**: Low | **Estimated**: 1 day

**Files to Create:**
- `frontend/src/routes/_layout/embeddings.tsx`
- `frontend/src/components/Embeddings/EmbeddingsGenerator.tsx`

**Backend APIs:**
- [x] `POST /api/v1/embeddings` - Generate embeddings

**UI Features:**
- [ ] Text input area
- [ ] Model selector
- [ ] Generate embeddings
- [ ] Display vector (collapsible)
- [ ] Copy vector to clipboard
- [ ] Download as JSON

---

### 2.4 File Management ✋ TODO
**Priority**: Medium | **Estimated**: 2 days

**Files to Create:**
- `frontend/src/routes/_layout/files.tsx`
- `frontend/src/components/Files/FileList.tsx`
- `frontend/src/components/Files/FileUpload.tsx`
- `frontend/src/components/Files/FileActionsMenu.tsx`

**Backend APIs:**
- [x] `POST /api/v1/files/` - Upload file
- [x] `GET /api/v1/files/` - List files
- [x] `DELETE /api/v1/files/{id}` - Delete file

**UI Features:**
- [ ] File upload (drag & drop)
- [ ] File list with metadata
- [ ] File type icons
- [ ] Download file
- [ ] Delete file
- [ ] File usage info

---

### 2.5 Batch Jobs ✋ TODO
**Priority**: Low | **Estimated**: 2 days

**Files to Create:**
- `frontend/src/routes/_layout/batches.tsx`
- `frontend/src/components/Batches/BatchList.tsx`
- `frontend/src/components/Batches/CreateBatch.tsx`
- `frontend/src/components/Batches/BatchDetails.tsx`

**Backend APIs:**
- [x] `POST /api/v1/batches/` - Create batch
- [x] `GET /api/v1/batches/` - List batches
- [x] `GET /api/v1/batches/{id}` - Get batch details
- [x] `DELETE /api/v1/batches/{id}` - Cancel batch

**UI Features:**
- [ ] Batch jobs table
- [ ] Create batch from file
- [ ] Batch progress indicator
- [ ] Batch results download
- [ ] Cancel running batch
- [ ] Batch cost estimation

---

### 2.6 Audio Transcription ✋ TODO
**Priority**: Low | **Estimated**: 1 day

**Files to Create:**
- `frontend/src/routes/_layout/audio.tsx`
- `frontend/src/components/Audio/AudioTranscription.tsx`

**Backend APIs:**
- [x] `POST /api/v1/audio/transcriptions` - Transcribe audio

**UI Features:**
- [ ] Audio file upload
- [ ] Model selector (whisper, etc.)
- [ ] Transcription progress
- [ ] Display transcription
- [ ] Copy/download transcription
- [ ] Language selection

---

## Phase 3: Dashboard & Analytics (Week 5)

### 3.1 Enhanced Dashboard ✋ TODO
**Priority**: Medium | **Estimated**: 2 days

**Files to Update:**
- `frontend/src/routes/_layout/index.tsx`
- `frontend/src/components/Dashboard/`

**Features to Add:**
- [ ] System metrics cards (CPU, memory, disk)
- [ ] Model usage chart (requests/day)
- [ ] Recent activity feed
- [ ] Quick action buttons
- [ ] Server status overview
- [ ] Agent status summary
- [ ] Storage usage by models

---

### 3.2 Items Page (Template) ✋ TODO
**Priority**: Low | **Estimated**: 1 day

**Files to Update:**
- `frontend/src/routes/_layout/items.tsx`
- `frontend/src/components/Items/`

**Purpose:**
- Generic CRUD template for future features
- Demonstration of best practices
- Can be repurposed for new entity types

---

## Phase 4: Polish & Optimization (Week 6)

### 4.1 UI/UX Improvements ✋ TODO
**Priority**: Medium | **Estimated**: 2 days

**Tasks:**
- [ ] Add loading skeletons for all pages
- [ ] Implement error boundaries
- [ ] Add keyboard shortcuts
- [ ] Improve mobile responsiveness
- [ ] Add tooltips to all buttons
- [ ] Implement search/filter on all tables
- [ ] Add export functionality to tables

---

### 4.2 Performance Optimization ✋ TODO
**Priority**: Medium | **Estimated**: 1 day

**Tasks:**
- [ ] Code splitting for routes
- [ ] Lazy load heavy components
- [ ] Optimize bundle size
- [ ] Implement React Query caching
- [ ] Add service worker for offline support
- [ ] Optimize images and assets

---

### 4.3 Testing ✋ TODO
**Priority**: High | **Estimated**: 3 days

**Tasks:**
- [ ] Unit tests for components
- [ ] Integration tests for features
- [ ] E2E tests with Playwright
- [ ] Accessibility testing
- [ ] Performance testing
- [ ] Cross-browser testing

---

## File Naming Conventions

```
components/
├── Entity/
│   ├── EntityList.tsx       # Main list/table view
│   ├── EntityActionsMenu.tsx # Actions dropdown
│   ├── AddEntity.tsx        # Create dialog
│   ├── EditEntity.tsx       # Edit dialog
│   ├── DeleteEntity.tsx     # Delete confirmation
│   ├── EntityDetails.tsx    # Details view
│   ├── EntityStatus.tsx     # Status indicator
│   └── columns.tsx          # Table column definitions
```

---

## Component Templates

### List Component Template
```tsx
import { useSuspenseQuery } from "@tanstack/react-query"
import { createFileRoute } from "@tanstack/react-router"
import { Suspense, useState } from "react"
import { EntityService } from "@/client"
import { DataTable } from "@/components/Common/DataTable"
import { columns } from "@/components/Entity/columns"
import { Button } from "@/components/ui/button"
import { Plus } from "lucide-react"
import { AddEntity } from "@/components/Entity/AddEntity"

function EntityTableContent() {
  const { data: entities } = useSuspenseQuery({
    queryFn: async () => (await EntityService.readEntities({ query: { skip: 0, limit: 100 } })).data,
    queryKey: ["entities"],
  })

  return <DataTable columns={columns} data={entities} />
}

function Entity() {
  const [isAddOpen, setIsAddOpen] = useState(false)

  return (
    <div className="flex flex-col gap-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Entities</h1>
          <p className="text-muted-foreground">Manage your entities</p>
        </div>
        <Button onClick={() => setIsAddOpen(true)}>
          <Plus className="mr-2 h-4 w-4" />
          Add Entity
        </Button>
      </div>
      <Suspense fallback={<div>Loading...</div>}>
        <EntityTableContent />
      </Suspense>
      <AddEntity isOpen={isAddOpen} onClose={() => setIsAddOpen(false)} />
    </div>
  )
}

export const Route = createFileRoute("/_layout/entities")({
  component: Entity,
})
```

---

## Priority Legend

- **High**: Critical for MVP, implement first
- **Medium**: Important but can wait, implement in phase 2-3
- **Low**: Nice to have, implement if time permits

---

## Status Legend

- ✅ Complete
- 🚧 In Progress
- ✋ TODO - Not started
- 📅 Scheduled

---

## Notes

- All new components should follow existing patterns in `Models/` folder
- Use Shadcn UI components for consistency
- Implement proper error handling with toast notifications
- Add loading states for all async operations
- Use React Query for data fetching and caching
- Follow accessibility best practices (ARIA labels, keyboard navigation)
- Write tests for new features
